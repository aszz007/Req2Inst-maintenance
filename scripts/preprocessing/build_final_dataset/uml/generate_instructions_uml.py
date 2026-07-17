"""Generate UML-domain crowdsourcing instructions with browser-assisted batching."""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
from datetime import datetime
import chardet
import json

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"dataset/uml"
GPT_URL = "https://sass-node1.chatshare.biz/"

CSV_FILE = "uml_dataset.csv"

BATCH_SIZE = 1
REFRESH_INTERVAL = 1
CHECK_INTERVAL = 100
TEST_MODE_LIMIT = 10

WAIT_NEW_RESPONSE_TIMEOUT = 60
CONTENT_STABLE_CHECKS = 3

DOMAIN_EXAMPLES = {
    "ecommerce": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Payment Gateway"}, {"name": "Inventory System"}],
  "use_cases": [
    {"name": "Place Order", "description": "Customer places an order"},
    {"name": "Verify Stock", "description": "Check product availability"},
    {"name": "Process Payment", "description": "Handle payment transaction"},
    {"name": "Send Confirmation", "description": "Email order confirmation"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Place Order"},
    {"type": "include", "from": "Place Order", "to": "Verify Stock"},
    {"type": "include", "from": "Place Order", "to": "Process Payment"},
    {"type": "extend", "from": "Place Order", "to": "Send Confirmation"}
  ],
  "overall_description": "E-commerce order placement system with mandatory stock verification and payment processing, plus optional email confirmation."
}""",
        "instruction": """Definition: In this task, implement the "Place Order" workflow where Customer interacts with the system, ensuring mandatory stock verification and payment processing steps are completed.
Emphasis & Caution: You MUST execute "Verify Stock" and "Process Payment" as required prerequisites (include relationships) before finalizing the order. "Send Confirmation" is a conditional extension that triggers upon successful order completion.
Things to Avoid: Do not use actor position metadata to determine business logic or workflow sequence. Do not implement UI layout based on position values."""
    },

    "authentication": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "OAuth Provider"}, {"name": "Email Service"}],
  "use_cases": [
    {"name": "Login", "description": "User authentication"},
    {"name": "Validate Credentials", "description": "Verify username and password"},
    {"name": "Generate Token", "description": "Create session token"},
    {"name": "Send Verification Email", "description": "Email verification for new devices"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Login"},
    {"type": "include", "from": "Login", "to": "Validate Credentials"},
    {"type": "include", "from": "Login", "to": "Generate Token"},
    {"type": "extend", "from": "Login", "to": "Send Verification Email"}
  ],
  "overall_description": "User authentication system with mandatory credential validation and token generation, plus optional email verification for new devices."
}""",
        "instruction": """Definition: In this task, implement the "Login" authentication workflow where User interacts with the system, ensuring mandatory credential validation and token generation.
Emphasis & Caution: You MUST enforce "Validate Credentials" and "Generate Token" as required steps (include relationships) that execute automatically during login. "Send Verification Email" is a conditional extension triggered when login occurs from a new device.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "content_management": {
        "json": """{
  "actors": [{"name": "Author"}, {"name": "Editor"}, {"name": "Publisher"}],
  "use_cases": [
    {"name": "Create Article", "description": "Author creates content"},
    {"name": "Submit for Review", "description": "Submit to editorial queue"},
    {"name": "Approve Content", "description": "Editor approves article"},
    {"name": "Publish", "description": "Make content live"}
  ],
  "relationships": [
    {"type": "association", "from": "Author", "to": "Create Article"},
    {"type": "include", "from": "Create Article", "to": "Submit for Review"},
    {"type": "association", "from": "Editor", "to": "Approve Content"},
    {"type": "extend", "from": "Approve Content", "to": "Publish"}
  ],
  "overall_description": "Content management system where authors create articles with mandatory review submission, editors approve, and optional immediate publishing."
}""",
        "instruction": """Definition: In this task, implement the "Create Article" workflow where Author creates content with mandatory review submission, and the "Approve Content" process where Editor reviews articles with optional publishing.
Emphasis & Caution: You MUST enforce "Submit for Review" as a required step (include relationship) that executes automatically after article creation. "Publish" is a conditional extension of approval that triggers when immediate publishing is selected.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "social_interaction": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Follower"}, {"name": "Notification Service"}],
  "use_cases": [
    {"name": "Create Post", "description": "User creates a post"},
    {"name": "Validate Content", "description": "Check for prohibited content"},
    {"name": "Notify Followers", "description": "Send notifications to followers"},
    {"name": "Generate Thumbnail", "description": "Create image preview"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Create Post"},
    {"type": "include", "from": "Create Post", "to": "Validate Content"},
    {"type": "extend", "from": "Create Post", "to": "Notify Followers"},
    {"type": "extend", "from": "Create Post", "to": "Generate Thumbnail"}
  ],
  "overall_description": "Social media post creation system with mandatory content validation and optional follower notifications and thumbnail generation."
}""",
        "instruction": """Definition: In this task, implement the "Create Post" workflow where User creates social media content with mandatory content validation.
Emphasis & Caution: You MUST enforce "Validate Content" as a required step (include relationship) that executes before post publication. "Notify Followers" and "Generate Thumbnail" are conditional extensions that trigger based on user preferences or content type.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "customer_service": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Support Agent"}, {"name": "Ticketing System"}],
  "use_cases": [
    {"name": "Create Ticket", "description": "Customer creates support ticket"},
    {"name": "Assign Category", "description": "Categorize the issue"},
    {"name": "Set Priority", "description": "Determine urgency level"},
    {"name": "Escalate Issue", "description": "Route to senior support"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Create Ticket"},
    {"type": "include", "from": "Create Ticket", "to": "Assign Category"},
    {"type": "include", "from": "Create Ticket", "to": "Set Priority"},
    {"type": "extend", "from": "Create Ticket", "to": "Escalate Issue"}
  ],
  "overall_description": "Customer support ticket system with mandatory categorization and priority setting, plus optional escalation for complex issues."
}""",
        "instruction": """Definition: In this task, implement the "Create Ticket" workflow where Customer submits support requests with mandatory category assignment and priority setting.
Emphasis & Caution: You MUST enforce "Assign Category" and "Set Priority" as required steps (include relationships) that execute during ticket creation. "Escalate Issue" is a conditional extension that triggers when the issue meets escalation criteria.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "data_analysis": {
        "json": """{
  "actors": [{"name": "Analyst"}, {"name": "Data Warehouse"}, {"name": "Reporting Engine"}],
  "use_cases": [
    {"name": "Run Analysis", "description": "Execute data analysis"},
    {"name": "Fetch Data", "description": "Retrieve data from warehouse"},
    {"name": "Generate Report", "description": "Create analysis report"},
    {"name": "Export to CSV", "description": "Export results to CSV"}
  ],
  "relationships": [
    {"type": "association", "from": "Analyst", "to": "Run Analysis"},
    {"type": "include", "from": "Run Analysis", "to": "Fetch Data"},
    {"type": "include", "from": "Run Analysis", "to": "Generate Report"},
    {"type": "extend", "from": "Run Analysis", "to": "Export to CSV"}
  ],
  "overall_description": "Data analysis system with mandatory data fetching and report generation, plus optional CSV export functionality."
}""",
        "instruction": """Definition: In this task, implement the "Run Analysis" workflow where Analyst executes data analysis with mandatory data fetching and report generation.
Emphasis & Caution: You MUST enforce "Fetch Data" and "Generate Report" as required steps (include relationships) that execute during analysis. "Export to CSV" is a conditional extension that triggers when the analyst requests data export.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "permission_management": {
        "json": """{
  "actors": [{"name": "Admin"}, {"name": "User"}, {"name": "Audit System"}],
  "use_cases": [
    {"name": "Assign Role", "description": "Assign role to user"},
    {"name": "Validate Permissions", "description": "Check role validity"},
    {"name": "Update Access Rights", "description": "Modify user permissions"},
    {"name": "Log Changes", "description": "Record permission changes"}
  ],
  "relationships": [
    {"type": "association", "from": "Admin", "to": "Assign Role"},
    {"type": "include", "from": "Assign Role", "to": "Validate Permissions"},
    {"type": "include", "from": "Assign Role", "to": "Update Access Rights"},
    {"type": "extend", "from": "Assign Role", "to": "Log Changes"}
  ],
  "overall_description": "Permission management system with mandatory permission validation and access rights updates, plus optional audit logging."
}""",
        "instruction": """Definition: In this task, implement the "Assign Role" workflow where Admin assigns roles to users with mandatory permission validation and access rights updates.
Emphasis & Caution: You MUST enforce "Validate Permissions" and "Update Access Rights" as required steps (include relationships) that execute during role assignment. "Log Changes" is a conditional extension that triggers when audit logging is enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "notification_system": {
        "json": """{
  "actors": [{"name": "System"}, {"name": "User"}, {"name": "Email Service"}, {"name": "SMS Gateway"}],
  "use_cases": [
    {"name": "Send Notification", "description": "Trigger notification"},
    {"name": "Check Preferences", "description": "Verify user notification settings"},
    {"name": "Send Email", "description": "Send email notification"},
    {"name": "Send SMS", "description": "Send SMS notification"}
  ],
  "relationships": [
    {"type": "association", "from": "System", "to": "Send Notification"},
    {"type": "include", "from": "Send Notification", "to": "Check Preferences"},
    {"type": "extend", "from": "Send Notification", "to": "Send Email"},
    {"type": "extend", "from": "Send Notification", "to": "Send SMS"}
  ],
  "overall_description": "Notification system with mandatory preference checking and optional email or SMS delivery based on user settings."
}""",
        "instruction": """Definition: In this task, implement the "Send Notification" workflow where System triggers notifications with mandatory preference checking.
Emphasis & Caution: You MUST enforce "Check Preferences" as a required step (include relationship) that executes before sending notifications. "Send Email" and "Send SMS" are conditional extensions that trigger based on user notification preferences.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "file_management": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Storage System"}, {"name": "Virus Scanner"}],
  "use_cases": [
    {"name": "Upload File", "description": "User uploads a file"},
    {"name": "Scan for Viruses", "description": "Check file for malware"},
    {"name": "Store File", "description": "Save file to storage"},
    {"name": "Generate Preview", "description": "Create file thumbnail"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Upload File"},
    {"type": "include", "from": "Upload File", "to": "Scan for Viruses"},
    {"type": "include", "from": "Upload File", "to": "Store File"},
    {"type": "extend", "from": "Upload File", "to": "Generate Preview"}
  ],
  "overall_description": "File upload system with mandatory virus scanning and storage, plus optional preview generation for supported file types."
}""",
        "instruction": """Definition: In this task, implement the "Upload File" workflow where User uploads files with mandatory virus scanning and storage operations.
Emphasis & Caution: You MUST enforce "Scan for Viruses" and "Store File" as required steps (include relationships) that execute during file upload. "Generate Preview" is a conditional extension that triggers for supported file types (images, documents).
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "booking_system": {
        "json": """{
  "actors": [{"name": "Customer"}, {"name": "Calendar System"}, {"name": "Payment Gateway"}],
  "use_cases": [
    {"name": "Book Appointment", "description": "Customer books a time slot"},
    {"name": "Check Availability", "description": "Verify slot availability"},
    {"name": "Process Payment", "description": "Handle booking payment"},
    {"name": "Send Reminder", "description": "Send appointment reminder"}
  ],
  "relationships": [
    {"type": "association", "from": "Customer", "to": "Book Appointment"},
    {"type": "include", "from": "Book Appointment", "to": "Check Availability"},
    {"type": "include", "from": "Book Appointment", "to": "Process Payment"},
    {"type": "extend", "from": "Book Appointment", "to": "Send Reminder"}
  ],
  "overall_description": "Appointment booking system with mandatory availability checking and payment processing, plus optional reminder notifications."
}""",
        "instruction": """Definition: In this task, implement the "Book Appointment" workflow where Customer books time slots with mandatory availability checking and payment processing.
Emphasis & Caution: You MUST enforce "Check Availability" and "Process Payment" as required steps (include relationships) that execute during booking. "Send Reminder" is a conditional extension that triggers when reminder notifications are enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    }
}

SYSTEM_PROMPT = """You are a software architecture and crowdsourcing task design expert. Based on the input UML Use Case Diagram structured data (JSON format), write an English task instruction for crowdsourcing workers.

Core Principles:
1. Data-Driven: Actor names and Use Case names in the instruction must strictly reference the original names from JSON source data. Do not omit, abbreviate, or rewrite.
2. Logic Priority, Visuals Secondary: Completely ignore visual layout information like position (e.g., top_left) in input data. Focus on parsing business logic in relationships.
3. Relationship Semantics Translation:
   - include -> Translate to "Mandatory step" or "Required prerequisite"
   - extend -> Translate to "Conditional flow" or "Optional"
   - association -> Translate to "Interaction" or "Access"
4. Structured Format: Strictly follow the three-part format defined below.

Output Format Requirements:
- Definition: Use a clear imperative sentence to describe the core system objective. Must start with "In this task,".
- Emphasis & Caution: Highlight mandatory flows (include) and conditional extension flows (extend). Use "-" if none.
- Things to Avoid: List prohibited operations (e.g., focusing on node positions, implementing UI styles). Use "-" if nothing specific.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must start with "In this task," and explicitly list actors and use cases from JSON data
- Translate relationship types (include/extend/association) to business logic terms
- Keep all sections concise
- Output ONLY these three lines, nothing else

Reference Example:
{example}

Please generate instructions for the following {count} UML use case diagram(s). Strictly follow the format below and do not add extra explanations:

{uml_data}

Output format (3 lines only):

Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...
"""

class GPTAutomator:
    """Automate browser-assisted prompt submission and response collection."""
    def __init__(self, test_mode=True):
        self.test_mode = test_mode
        self.driver = None
        self.current_tab = None
        self.processed_count = 0
        self.error_log = []

        self.cached_input_selector = None
        self.cached_button_selector = None

        self.response_count_before_send = 0

        self.batches_since_refresh = 0

    def init_driver(self):
        """Initialize the browser driver."""
        print("\n" + "="*60)
        print("Initializing browser...")
        print("="*60)

        if not os.path.exists(CHROME_PATH):
            raise FileNotFoundError(f"Chrome executable not found at: {CHROME_PATH}")
        print(f" Chrome path validated successfully")

        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')

            user_data_dir = os.path.join(os.getcwd(), 'chrome_user_data')
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
            options.add_argument(f'--user-data-dir={user_data_dir}')

            options.add_argument('--disable-extensions')
            options.add_argument('--remote-debugging-port=9222')
            options.add_argument('--start-maximized')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)

            print(" ChromeOptions configuration complete")
            print("Starting ChromeDriver...")
            self.driver = webdriver.Chrome(options=options)
            print(f" ChromeDriver started successfully")

            print(f"\nNavigating to: {GPT_URL}")
            self.driver.get(GPT_URL)
            time.sleep(8)

            print(f" Page loaded: {self.driver.title}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n Browser initialization failed: {e}")
            raise

    def extract_domain_from_header(self, header: str) -> str:
        """Extract domain from header."""
        header = header.lower()

        known_domains = [
            "ecommerce", "authentication", "content_management",
            "social_interaction", "customer_service", "data_analysis",
            "permission_management", "notification_system",
            "file_management", "booking_system"
        ]

        for domain in sorted(known_domains, key=len, reverse=True):
            if domain in header:
                return domain

        return "unknown"

    def get_example_for_domain(self, domain: str) -> str:
        """Return example for domain."""
        if domain not in DOMAIN_EXAMPLES:
            domain = "authentication"

        example_data = DOMAIN_EXAMPLES[domain]
        example_text = f"{example_data['json']}\n\nOutput Instruction:\n{example_data['instruction']}"

        return example_text

    def normalize_three_part_format(self, text):
        """Normalize three part format."""
        if not text:
            return text

        keywords = ['Definition:', 'Emphasis & Caution:', 'Things to Avoid:']

        all_present = all(keyword in text for keyword in keywords)
        if not all_present:
            return text

        result = text
        for keyword in keywords:
            parts = result.split(keyword)
            if len(parts) > 1:
                normalized_parts = []
                for i, part in enumerate(parts):
                    if i == 0:
                        normalized_parts.append(part.rstrip())
                    else:
                        if normalized_parts:
                            normalized_parts.append('\n' + keyword + part)
                        else:
                            normalized_parts.append(keyword + part)
                result = ''.join(normalized_parts)

        return result.strip()

    def clean_json_data(self, json_str):
        """Clean JSON data."""
        try:
            data = json.loads(json_str)

            if 'actors' in data and isinstance(data['actors'], list):
                for actor in data['actors']:
                    if 'position' in actor:
                        del actor['position']


            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  JSON cleanup failed: {e}")
            return json_str

    def find_input_box(self, debug=False):
        """Find input box."""
        if debug:
            print(" Locating input field...")

        if self.cached_input_selector:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.cached_input_selector))
                )
                if element.is_displayed() and element.is_enabled():
                    if debug:
                        print(f"  Cached selector used successfully")
                    return element
                else:
                    self.cached_input_selector = None
            except:
                self.cached_input_selector = None

        selectors = [
            ("CSS", "div[contenteditable='true']"),
            ("CSS", "[contenteditable='true']"),
            ("CSS", "textarea"),
            ("CSS", "textarea[placeholder*='询问']"),
            ("CSS", "form textarea"),
            ("CSS", "div[class*='input'] textarea"),
        ]

        for selector_type, selector in selectors:
            try:
                if debug:
                    print(f"  Attempt: {selector}")

                element = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )

                if element.is_displayed() and element.is_enabled():
                    self.cached_input_selector = selector
                    if debug:
                        print(f"  Succeeded: {selector}")
                    return element

            except:
                continue

        raise NoSuchElementException("Unable to locate the input field")

    def find_submit_button(self):
        """Find submit button."""
        if self.cached_button_selector:
            try:
                button = self.driver.find_element(By.CSS_SELECTOR, self.cached_button_selector)
                if button.is_displayed() and button.is_enabled():
                    return button
                else:
                    self.cached_button_selector = None
            except:
                self.cached_button_selector = None

        selectors = [
            "button[data-testid='send-button']",
            "button[type='submit']",
            "button:has(svg)",
            "button[aria-label*='Send']",
            "button[aria-label*='发送']",
        ]

        for selector in selectors:
            try:
                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        self.cached_button_selector = selector
                        return button
            except:
                continue

        return None

    def get_current_response_count(self):
        """Return current response count."""
        try:
            response_selectors = [
                "div[data-message-author-role='assistant']",
                "article[data-turn='assistant']",
                "article[data-testid*='conversation-turn'] div.markdown.prose",
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    if selector == "article[data-testid*='conversation-turn'] div.markdown.prose":
                        valid_count = 0
                        for elem in elements:
                            try:
                                parent_article = self.driver.execute_script(
                                    "return arguments[0].closest('article')",
                                    elem
                                )
                                if parent_article:
                                    turn_type = parent_article.get_attribute('data-turn')
                                    if turn_type == 'assistant':
                                        valid_count += 1
                            except:
                                continue
                        if valid_count > 0:
                            return valid_count
                    else:
                        if elements and len(elements) > 0:
                            return len(elements)
                except:
                    continue
            return 0
        except:
            return 0

    def check_response_still_updating(self):
        """Check response still updating."""
        try:
            selector = "div[data-message-author-role='assistant']"

            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            current_count = len(elements)

            if current_count <= self.response_count_before_send:
                return True

            if not elements:
                return True

            last_response = elements[-1]
            first_text = last_response.text
            first_len = len(first_text)

            time.sleep(0.8)

            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            if len(elements) > 0:
                last_response = elements[-1]
                second_text = last_response.text
                second_len = len(second_text)

                if second_len > first_len:
                    return True
                return False

            return False
        except Exception as e:
            print(f"Error checking for updates: {e}")
            return False

    def wait_for_response_complete(self, timeout=300):
        """Wait for the browser response to finish."""
        print("  Waiting for generation...", end='', flush=True)
        start_time = time.time()
        last_progress_time = start_time

        print(" [Waiting for response]", end='', flush=True)
        response_appeared = False

        consecutive_validation_failures = 0
        MAX_VALIDATION_FAILURES = 50

        check_count = 0
        while time.time() - start_time < WAIT_NEW_RESPONSE_TIMEOUT:
            try:
                current_count = self.get_current_response_count()

                if current_count > self.response_count_before_send:
                    print(f" [Possible new response detected; validating]", end='', flush=True)
                    time.sleep(2)

                    recheck_count = self.get_current_response_count()

                    if recheck_count > self.response_count_before_send:
                        if self._validate_new_response():
                            elapsed = int(time.time() - start_time)
                            response_appeared = True
                            print(f"  [New response confirmed; elapsed {elapsed}s]", end='', flush=True)
                            break
                        else:
                            consecutive_validation_failures += 1
                            print(f" [Content validation failed{consecutive_validation_failures}/{MAX_VALIDATION_FAILURES}]", end='',
                                  flush=True)

                            if consecutive_validation_failures >= MAX_VALIDATION_FAILURES:
                                elapsed = int(time.time() - start_time)
                                print(f"  [Validation failed but accepted; elapsed {elapsed}s]", end='', flush=True)
                                response_appeared = True
                                break

                            time.sleep(2)
                    else:
                        print(f" [Count not stable; continuing to wait]", end='', flush=True)
                        consecutive_validation_failures = 0
                        time.sleep(1)
                else:
                    consecutive_validation_failures = 0

                check_count += 1
                if check_count % 4 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"[{elapsed}s]", end='', flush=True)

            except Exception as e:
                print(f"![{str(e)[:20]}]", end='', flush=True)

            time.sleep(0.5)

        if not response_appeared:
            print(f"  Response wait timed out ({WAIT_NEW_RESPONSE_TIMEOUT}s)")
            return False

        time.sleep(1)
        print(" [Check complete]", end='', flush=True)

        stable_count = 0
        max_stability_checks = 10

        for check_round in range(max_stability_checks):
            try:
                is_updating = self.check_response_still_updating()

                if is_updating:
                    stable_count = 0
                    print(".", end='', flush=True)
                else:
                    stable_count += 1
                    if stable_count >= CONTENT_STABLE_CHECKS:
                        print("  Complete")
                        return True
                    else:
                        print(".", end='', flush=True)

                current_time = time.time()
                if current_time - last_progress_time >= 5:
                    total_elapsed = int(current_time - start_time)
                    print(f"[{total_elapsed}s]", end='', flush=True)
                    last_progress_time = current_time

                time.sleep(1)

            except Exception as e:
                print(f" [{str(e)[:20]}]", end='', flush=True)
                time.sleep(1)

        print("  Completed (reached check limit)")
        return True

    def extract_response(self):
        """Extract response."""
        try:
            selector = "div[data-message-author-role='assistant']"

            response_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            current_count = len(response_elements)

            if current_count > self.response_count_before_send:
                last_response = response_elements[-1]

                try:
                    markdown_div = last_response.find_element(
                        By.CSS_SELECTOR,
                        "div.markdown.prose"
                    )
                    response_text = markdown_div.text
                except:
                    response_text = last_response.text

                if response_text and len(response_text) > 10:
                    print(f"  Extracted response ({len(response_text)} characters)")

                    has_definition = "Definition:" in response_text
                    has_emphasis = "Emphasis" in response_text or "Caution" in response_text
                    has_avoid = "Avoid" in response_text

                    if has_definition or has_emphasis or has_avoid:
                        print(f"  Content validation passed (contains instruction keywords)")
                    else:
                        print(f"  Warning: response may not contain the expected format")

                    return response_text
                else:
                    print(f"  Extracted content is too short: {len(response_text) if response_text else 0} characters")

            print(f"  Unable to extract a valid response (current {current_count} items, before sending {self.response_count_before_send} items)")
            return ""

        except Exception as e:
            print(f"  Failed to extract response: {e}")
            return ""

    def _validate_new_response(self):
        """Validate new response."""
        try:
            selector = "div[data-message-author-role='assistant']"
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

            if not elements or len(elements) <= self.response_count_before_send:
                return False

            last_response = elements[-1]

            try:
                markdown_div = last_response.find_element(
                    By.CSS_SELECTOR,
                    "div.markdown.prose"
                )
                text = markdown_div.text.strip()
            except:
                text = last_response.text.strip()

            if len(text) < 5:
                return False

            if len(text) <= 3 and text in ["●", "⚫", "🔴", "...", "•"]:
                return False

            has_content = any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in text)
            if not has_content and len(text) < 20:
                return False

            return True

        except Exception as e:
            print(f"[Validation exception, accepted]", end='', flush=True)
            return True

    def parse_instructions(self, response_text, expected_count):
        """Parse instructions."""
        instructions = []

        pattern = r'【图像\d+】\s*\n(.*?)(?=【图像\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == expected_count:
            for match in matches:
                instruction = match.strip()
                instruction = self.normalize_three_part_format(instruction)
                instructions.append(instruction)
        else:
            parts = response_text.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instruction = 'Definition:' + part.strip()
                    instruction = self.normalize_three_part_format(instruction)
                    instructions.append(instruction)

        return instructions

    def send_prompt(self, prompt_text, max_retries=3):
        """Submit a prompt to the browser session."""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"\n Sending prompt...")
                    self.response_count_before_send = self.get_current_response_count()
                    print(f"  Current page has {self.response_count_before_send} responses")
                else:
                    print(f"  Retry {attempt}/{max_retries-1}...")

                input_box = self.find_input_box(debug=(attempt == 0))
                if not input_box:
                    if attempt < max_retries - 1:
                        self.driver.refresh()
                        time.sleep(8)
                        continue
                    return False

                self.driver.execute_script("arguments[0].focus();", input_box)
                time.sleep(0.3)

                tag_name = input_box.tag_name.lower()
                if tag_name == "textarea":
                    self.driver.execute_script("arguments[0].value = '';", input_box)
                else:
                    self.driver.execute_script("arguments[0].textContent = '';", input_box)

                time.sleep(0.3)

                if tag_name == "textarea":
                    self.driver.execute_script("""
                        var elem = arguments[0];
                        var text = arguments[1];
                        elem.value = text;
                        elem.dispatchEvent(new Event('input', { bubbles: true }));
                    """, input_box, prompt_text)
                else:
                    self.driver.execute_script("""
                        var elem = arguments[0];
                        var text = arguments[1];
                        elem.textContent = text;
                        elem.dispatchEvent(new Event('input', { bubbles: true }));
                        elem.focus();
                    """, input_box, prompt_text)

                time.sleep(1)

                current_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not current_value or len(current_value) < 100:
                    if attempt < max_retries - 1:
                        continue
                    return False

                print(f"  Text set successfully ({len(current_value)} characters)")

                button = self.find_submit_button()
                if button:
                    self.driver.execute_script("arguments[0].click();", button)
                    print(f"  Clicked the Send button")
                else:
                    input_box.send_keys(Keys.RETURN)
                    print(f"  Sent with Enter")

                time.sleep(2)

                check_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not check_value or len(check_value.strip()) < 50:
                    print("  Confirmed message sent")
                    return True

                time.sleep(2)
                return True

            except Exception as e:
                print(f"  Send error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return False

        return False

    def process_batch(self, uml_data_batch, start_idx):
        """Process batch."""
        print(f"\n{'=' * 60}")
        print(f"Processing record {start_idx + 1}-{start_idx + len(uml_data_batch)} UML records")
        print(f"{'=' * 60}")

        first_header = uml_data_batch[0][0] if uml_data_batch else ""
        domain = self.extract_domain_from_header(first_header)
        example_text = self.get_example_for_domain(domain)

        print(f"  Detected domain: {domain}")
        print(f"  Example: {domain} domain\n")

        data_text = ""
        for i, (header, description) in enumerate(uml_data_batch, 1):
            cleaned_json = self.clean_json_data(description)
            data_text += f"{i}. [UML Diagram: {header}]\n{cleaned_json}\n\n"

        prompt = SYSTEM_PROMPT.format(
            example=example_text,
            count=len(uml_data_batch),
            uml_data=data_text
        )

        max_retries = 3
        response = None
        retry_happened = False

        for retry_count in range(max_retries):
            if retry_count > 0:
                retry_happened = True
                print(f"\n Generation error detected; retrying ({retry_count}/{max_retries - 1})...")
                print("  Opening a new conversation to retry...")
                self.start_new_chat()
                time.sleep(2)

            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': '发送失败'
                    })
                    return [None] * len(uml_data_batch), retry_happened

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': '等待超时'
                    })
                    return [None] * len(uml_data_batch), retry_happened

            response = self.extract_response()
            print(f"\nResponse preview: {response[:200]}...\n")

            error_keywords = [
                "Something went wrong",
                "生成响应时出错",
                "出现错误",
                "error occurred",
                "failed to generate",
                "请尝试等待一会儿",
                "新建一个对话"
            ]

            is_error_response = any(keyword in response for keyword in error_keywords)

            if is_error_response:
                print(f"  Generation error detected: {response[:100]}")
                if retry_count < max_retries - 1:
                    print(f"  Will resend in 3 seconds...")
                    continue
                else:
                    print(f"  Maximum retry count reached ({max_retries}),abandoning this batch")
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(uml_data_batch)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(uml_data_batch), retry_happened
            else:
                print(f"  Response looks normal; preparing to parse")
                break

        instructions = self.parse_instructions(response, len(uml_data_batch))

        if len(instructions) != len(uml_data_batch):
            print(f"  Warning: expected {len(uml_data_batch)} items, actual {len(instructions)} items")
            while len(instructions) < len(uml_data_batch):
                instructions.append(None)
            instructions = instructions[:len(uml_data_batch)]

        return instructions, retry_happened

    def start_new_chat(self):
        """Start a new browser chat."""
        print("\n>>> Opening a new conversation...")
        try:
            from selenium.webdriver.common.action_chains import ActionChains

            print("  Sending shortcut Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  Shortcut sent")

            time.sleep(3)

            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            self.batches_since_refresh = 0

            print("  New conversation ready\n")

        except Exception as e:
            print(f"  Failed to open a new conversation: {e}")
            print("  Continuing in the current conversation")

    def process_file(self, csv_path):
        """Process file."""
        print(f"\n{'#' * 60}")
        print(f"# Processing file: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

        self.start_new_chat()

        try:
            with open(csv_path, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"File encoding: {encoding}")

            try:
                df = pd.read_csv(csv_path, encoding=encoding)
            except:
                for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc)
                        print(f"  Using {enc} encoding")
                        break
                    except:
                        continue
                else:
                    raise Exception("Failed to read the file")

        except Exception as e:
            print(f" Failed to read file: {e}")
            return 0

        required_columns = ['Header', 'Description', 'Instruction']
        if not all(col in df.columns for col in required_columns):
            print(f" CSV file is missing required columns: {required_columns}")
            print(f"  Current columns: {df.columns.tolist()}")
            return 0

        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        total_rows = len(df)

        if self.test_mode:
            print(f"*** Test mode: randomly select one item per domain ***\n")

            df['domain'] = df['Header'].apply(lambda h: self.extract_domain_from_header(h))

            domain_counts = df['domain'].value_counts()
            print("Domain distribution:")
            for domain, count in domain_counts.items():
                print(f"  {domain}: {count} items")

            selected_indices = []
            for domain in domain_counts.index:
                domain_df = df[df['domain'] == domain]
                if len(domain_df) > 0:
                    sampled = domain_df.sample(n=1, random_state=None)
                    selected_indices.extend(sampled.index.tolist())

            selected_indices.sort()

            print(f"\nSelected {len(selected_indices)} data items for testing:")
            for idx in selected_indices:
                header = df.loc[idx, 'Header']
                domain = df.loc[idx, 'domain']
                print(f"  - {header} (domain: {domain})")
            print()

            df = df.loc[selected_indices].reset_index(drop=True)
            total_rows = len(df)

        print(f"Total to process: {total_rows} UML records\n")
        print(f"Refresh strategy: every {REFRESH_INTERVAL} items ({REFRESH_INTERVAL//BATCH_SIZE} batches) or refresh the conversation after a retry\n")

        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)

            batch_data = []
            for idx in range(i, batch_end):
                header = df.loc[idx, 'Header']
                description = df.loc[idx, 'Description']
                batch_data.append((header, description))

            instructions, retry_happened = self.process_batch(batch_data, i)

            for j, instruction in enumerate(instructions):
                if instruction:
                    instruction = self.normalize_three_part_format(instruction)
                    df.at[i + j, 'Instruction'] = instruction
                else:
                    df.at[i + j, 'Instruction'] = "ERROR: 生成失败"

            self.processed_count += len(instructions)
            self.batches_since_refresh += 1

            if (i + BATCH_SIZE) < total_rows:
                print(f"\n Batch complete; refreshed - processed {self.batches_since_refresh} batch ({self.batches_since_refresh * BATCH_SIZE} data items)")
                print(f"  Current progress: {i + BATCH_SIZE}/{total_rows}")
                self.start_new_chat()

            if (i + BATCH_SIZE) % 50 == 0:
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"\n Progress saved: {i + BATCH_SIZE}/{total_rows}\n")

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n File processing complete: {os.path.basename(csv_path)}")
        print(f"  Processed {total_rows} UML records\n")

        return total_rows

    def run(self):
        """Run the workflow."""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'Batch UML business-logic instruction generation system':^60}")
        print(f"{'=' * 60}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'Test mode (one item per domain, 10 items total)' if self.test_mode else 'Full mode'}")
        print(f"Batch size: {BATCH_SIZE} items/batch")
        print(f"Refresh interval: {REFRESH_INTERVAL} items ({REFRESH_INTERVAL//BATCH_SIZE} batches)")
        print(f"Response timeout: {WAIT_NEW_RESPONSE_TIMEOUT} seconds")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_processed = self.process_file(csv_path)
            else:
                print(f" File not found: {csv_path}")
                total_processed = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'Processing complete':^60}")
            print(f"{'=' * 60}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed time: {duration}")
            print(f"Total processed: {total_processed} UML records")
            print(f"Succeeded: {total_processed - len(self.error_log)} items")
            print(f"Failed: {len(self.error_log)} items")

            if self.error_log:
                print(f"\nError log:")
                for error in self.error_log:
                    print(f"  - {error['range']}: {error['error']}")

            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"\n Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.driver:
                input("按 Enter 关闭浏览器...")
                self.driver.quit()
                print(" Browser closed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Select run mode:")
    print(f"  1. Test mode (one random item per domain, 10 items total)")
    print("  2. Full mode (process all data)")
    print("="*60)

    while True:
        choice = input("\n请输入选项 (1 或 2): ").strip()
        if choice == "1":
            test_mode = True
            print("\n Selected: test mode")
            break
        elif choice == "2":
            test_mode = False
            print("\n Selected: full mode")
            confirm = input(" 完整模式将处理大量数据,确认继续? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                print("Cancelled")
                exit(0)
        else:
            print(" Invalid option; enter 1 or 2")

    automator = GPTAutomator(test_mode=test_mode)
    automator.run()
