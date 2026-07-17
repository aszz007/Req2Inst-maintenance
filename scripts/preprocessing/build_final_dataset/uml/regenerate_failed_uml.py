"""Repair failed UML-domain instruction batches."""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
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
WAIT_NEW_RESPONSE_TIMEOUT = 60
CONTENT_STABLE_CHECKS = 3
ENABLE_PERIOD_CHECK = False

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
  "overall_description": "Data analysis system with mandatory data fetching and report generation, plus optional CSV export."
}""",
        "instruction": """Definition: In this task, implement the "Run Analysis" workflow where Analyst executes data analysis with mandatory data retrieval and report generation.
Emphasis & Caution: You MUST enforce "Fetch Data" and "Generate Report" as required steps (include relationships) that execute automatically during analysis. "Export to CSV" is a conditional extension that triggers when export is requested.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "booking_reservation": {
        "json": """{
  "actors": [{"name": "Guest"}, {"name": "Hotel System"}, {"name": "Payment Service"}],
  "use_cases": [
    {"name": "Book Room", "description": "Guest books a room"},
    {"name": "Check Availability", "description": "Verify room availability"},
    {"name": "Process Payment", "description": "Handle payment"},
    {"name": "Send Confirmation", "description": "Email booking confirmation"}
  ],
  "relationships": [
    {"type": "association", "from": "Guest", "to": "Book Room"},
    {"type": "include", "from": "Book Room", "to": "Check Availability"},
    {"type": "include", "from": "Book Room", "to": "Process Payment"},
    {"type": "extend", "from": "Book Room", "to": "Send Confirmation"}
  ],
  "overall_description": "Hotel booking system with mandatory availability check and payment, plus optional email confirmation."
}""",
        "instruction": """Definition: In this task, implement the "Book Room" workflow where Guest makes reservations with mandatory availability verification and payment processing.
Emphasis & Caution: You MUST enforce "Check Availability" and "Process Payment" as required steps (include relationships) before confirming booking. "Send Confirmation" is a conditional extension triggered upon successful booking.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "file_management": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Storage System"}, {"name": "Backup Service"}],
  "use_cases": [
    {"name": "Upload File", "description": "User uploads a file"},
    {"name": "Scan for Viruses", "description": "Check file safety"},
    {"name": "Store File", "description": "Save to storage"},
    {"name": "Create Backup", "description": "Backup the file"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Upload File"},
    {"type": "include", "from": "Upload File", "to": "Scan for Viruses"},
    {"type": "include", "from": "Upload File", "to": "Store File"},
    {"type": "extend", "from": "Upload File", "to": "Create Backup"}
  ],
  "overall_description": "File upload system with mandatory virus scanning and storage, plus optional backup creation."
}""",
        "instruction": """Definition: In this task, implement the "Upload File" workflow where User uploads files with mandatory virus scanning and storage.
Emphasis & Caution: You MUST enforce "Scan for Viruses" and "Store File" as required steps (include relationships) during upload. "Create Backup" is a conditional extension that triggers based on file importance or user settings.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "notification_system": {
        "json": """{
  "actors": [{"name": "System"}, {"name": "User"}, {"name": "Email Service"}],
  "use_cases": [
    {"name": "Send Notification", "description": "System sends notification"},
    {"name": "Format Message", "description": "Format notification content"},
    {"name": "Deliver to User", "description": "Send to user device"},
    {"name": "Log Activity", "description": "Record notification history"}
  ],
  "relationships": [
    {"type": "association", "from": "System", "to": "Send Notification"},
    {"type": "include", "from": "Send Notification", "to": "Format Message"},
    {"type": "include", "from": "Send Notification", "to": "Deliver to User"},
    {"type": "extend", "from": "Send Notification", "to": "Log Activity"}
  ],
  "overall_description": "Notification system with mandatory message formatting and delivery, plus optional activity logging."
}""",
        "instruction": """Definition: In this task, implement the "Send Notification" workflow where System sends notifications with mandatory message formatting and delivery.
Emphasis & Caution: You MUST enforce "Format Message" and "Deliver to User" as required steps (include relationships) before completing notification. "Log Activity" is a conditional extension that triggers when logging is enabled.
Things to Avoid: Do not use actor position values to determine business logic or workflow sequence. Do not implement UI layout based on position metadata."""
    },

    "access_control": {
        "json": """{
  "actors": [{"name": "User"}, {"name": "Admin"}, {"name": "Access Control System"}],
  "use_cases": [
    {"name": "Request Access", "description": "User requests resource access"},
    {"name": "Verify Identity", "description": "Authenticate user"},
    {"name": "Check Permissions", "description": "Verify user permissions"},
    {"name": "Log Access", "description": "Record access attempt"}
  ],
  "relationships": [
    {"type": "association", "from": "User", "to": "Request Access"},
    {"type": "include", "from": "Request Access", "to": "Verify Identity"},
    {"type": "include", "from": "Request Access", "to": "Check Permissions"},
    {"type": "extend", "from": "Request Access", "to": "Log Access"}
  ],
  "overall_description": "Access control system with mandatory identity verification and permission checking, plus optional access logging."
}""",
        "instruction": """Definition: In this task, implement the "Request Access" workflow where User requests resource access with mandatory identity verification and permission checks.
Emphasis & Caution: You MUST enforce "Verify Identity" and "Check Permissions" as required steps (include relationships) before granting access. "Log Access" is a conditional extension that triggers when audit logging is enabled.
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

"""


class UMLBatchRepairer:
    """Repair failed UML-instruction batches."""

    def __init__(self):
        self.driver = None
        self.repaired_count = 0
        self.error_log = []
        self.error_details = []

        self.cached_input_selector = None
        self.cached_button_selector = None

        self.response_count_before_send = 0

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

            user_data_dir = os.path.join(os.getcwd(), 'chrome_user_data_uml_repair')
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
            options.add_argument(f'--user-data-dir={user_data_dir}')

            options.add_argument('--disable-extensions')
            options.add_argument('--remote-debugging-port=9224')
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

    def find_input_box(self, debug=False):
        """Find input box."""
        if self.cached_input_selector:
            try:
                input_box = self.driver.find_element(By.CSS_SELECTOR, self.cached_input_selector)
                if input_box.is_displayed() and input_box.is_enabled():
                    return input_box
                else:
                    self.cached_input_selector = None
            except:
                self.cached_input_selector = None

        selectors = [
            "textarea[placeholder*='Message']",
            "textarea[data-id='root']",
            "div[contenteditable='true']",
            "textarea",
        ]

        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if debug and elements:
                    print(f"  Found {len(elements)} elements: {selector}")

                for elem in elements:
                    if elem.is_displayed() and elem.is_enabled():
                        self.cached_input_selector = selector
                        if debug:
                            print(f"  Input field located successfully: {selector}")
                        return elem
            except:
                continue

        if debug:
            print("  No usable input field found")
        return None

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

            print("  New conversation ready\n")

        except Exception as e:
            print(f"  Failed to open a new conversation: {e}")
            print("  Continuing in the current conversation")

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

    def _clean_and_merge_content(self, text):
        """Clean and merge content."""
        if not text:
            return ""

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line == '-':
                cleaned_lines.append(line)
            else:
                line = re.sub(r'^[•\-\*]\s+', '', line)
                line = line.strip()
                if line:
                    cleaned_lines.append(line)

        if len(cleaned_lines) > 1:
            result = '; '.join(cleaned_lines)
        elif len(cleaned_lines) == 1:
            result = cleaned_lines[0]
        else:
            result = ""

        if result != '-':
            result = re.sub(r'\s+', ' ', result).strip()

        return result

    def normalize_three_part_format(self, text):
        """Normalize three part format."""
        if not text:
            return text

        if 'Definition:' not in text or 'Emphasis & Caution:' not in text or 'Things to Avoid:' not in text:
            return text

        def_pos = text.find('Definition:')
        emp_pos = text.find('Emphasis & Caution:')
        avoid_pos = text.find('Things to Avoid:')

        if not (def_pos < emp_pos < avoid_pos):
            return text

        def_text = text[def_pos + len('Definition:'):emp_pos].strip()
        emp_text = text[emp_pos + len('Emphasis & Caution:'):avoid_pos].strip()
        avoid_text = text[avoid_pos + len('Things to Avoid:'):].strip()

        def_content = self._clean_and_merge_content(def_text)
        emp_content = self._clean_and_merge_content(emp_text)
        avoid_content = self._clean_and_merge_content(avoid_text)

        result = f"Definition: {def_content}\nEmphasis & Caution: {emp_content}\nThings to Avoid: {avoid_content}"
        return result

    def parse_uml_instruction(self, response_text):
        """Parse UML instruction."""
        pattern = r'【图像\d+】\s*\n(.*?)(?=【图像\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == 1:
            instruction = matches[0].strip()
            return self.normalize_three_part_format(instruction)

        parts = response_text.split('Definition:')
        for part in parts[1:]:
            if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                instruction = 'Definition:' + part.strip()
                return self.normalize_three_part_format(instruction)

        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

        definition = None
        emphasis = None
        avoid = None

        for line in lines:
            if line.startswith('Definition:'):
                definition = line
            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                emphasis = line
            elif line.startswith('Things to Avoid:'):
                avoid = line

        if definition and emphasis and avoid:
            instruction = f"{definition}\n{emphasis}\n{avoid}"
            return self.normalize_three_part_format(instruction)
        else:
            return None

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

    def validate_instruction_format(self, instruction):
        """Validate instruction format."""
        errors = []

        if not instruction or instruction.strip() == "":
            return False, ["指令为空"]

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            return False, [f"行数不足(期望3行,实际{len(lines)}行)"]

        has_definition = False
        has_emphasis = False
        has_avoid = False

        for line in lines:
            if line.startswith('Definition:'):
                has_definition = True
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition未以'In this task'开头")
                if ENABLE_PERIOD_CHECK and not content.endswith('.'):
                    errors.append("Definition缺少结尾句号")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if not content:
                    errors.append("Emphasis & Caution内容为空")
                if ENABLE_PERIOD_CHECK and content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution缺少结尾句号")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                if content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid未以句号结尾（可能生成不完整）")

        if not has_definition:
            errors.append("缺少Definition部分")
        if not has_emphasis:
            errors.append("缺少Emphasis & Caution部分")
        if not has_avoid:
            errors.append("缺少Things to Avoid部分")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def detect_error_batches(self, df):
        """Detect error batches."""
        print("\n" + "="*60)
        print("Starting error-data scan...")
        print("="*60)

        error_batches = []
        self.error_details = []

        total_rows = len(df)
        error_count = 0

        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)
            batch_has_error = False
            batch_error_details = []

            for idx in range(i, batch_end):
                instruction = str(df.loc[idx, 'Instruction'])
                header = df.loc[idx, 'Header']
                row_num = idx + 1

                if 'ERROR' in instruction.upper() and '生成失败' in instruction:
                    batch_has_error = True
                    error_msg = "含有ERROR标记"
                    batch_error_details.append((row_num, header, error_msg))
                    continue

                if not instruction or instruction.strip() == '' or instruction == 'nan':
                    batch_has_error = True
                    error_msg = "指令为空"
                    batch_error_details.append((row_num, header, error_msg))
                    continue

                is_valid, format_errors = self.validate_instruction_format(instruction)
                if not is_valid:
                    batch_has_error = True
                    error_msg = "; ".join(format_errors)
                    batch_error_details.append((row_num, header, error_msg))

            if batch_has_error:
                error_count += len(batch_error_details)
                self.error_details.extend(batch_error_details)

                batch_data = []
                for idx in range(i, batch_end):
                    header = df.loc[idx, 'Header']
                    description = df.loc[idx, 'Description']
                    batch_data.append((header, description))

                error_batches.append((len(error_batches) + 1, i, batch_data))

        print(f"\nScan results:")
        print(f"  Total data items: {total_rows}")
        print(f"  Erroneous data items: {error_count}")
        print(f"  Batches requiring repair: {len(error_batches)}")

        return error_batches

    def print_error_report(self):
        """Print error report."""
        if not self.error_details:
            return

        print("\n" + "="*60)
        print("Detailed error report")
        print("="*60)

        for row_num, header, error_msg in self.error_details:
            print(f"\nRow {row_num} item:")
            print(f"  Header: {header[:50]}...")
            print(f"  Error: {error_msg}")

        print("\n" + "="*60)

    def process_batch(self, batch_data, start_idx, batch_num, max_retries=3):
        """Process batch."""
        print(f"\n{'='*60}")
        print(f"Batch #{batch_num} | Data range: {start_idx + 1}-{start_idx + len(batch_data)}")
        print(f"{'='*60}")

        first_header = batch_data[0][0] if batch_data else ""
        domain = self.extract_domain_from_header(first_header)
        example_text = self.get_example_for_domain(domain)

        print(f"  Detected domain: {domain}")
        print(f"  Example: {domain} domain\n")

        data_text = ""
        for i, (header, description) in enumerate(batch_data, 1):
            cleaned_json = self.clean_json_data(description)
            data_text += f"{i}. [UML Diagram: {header}]\n{cleaned_json}\n\n"

        prompt = SYSTEM_PROMPT.format(
            example=example_text,
            count=len(batch_data),
            uml_data=data_text
        )

        for retry_count in range(max_retries):
            if retry_count > 0:
                print(f"\n Generation error detected; retrying ({retry_count}/{max_retries - 1})...")
                print("  Opening a new conversation to retry...")
                self.start_new_chat()
                time.sleep(2)

            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': '发送失败'
                    })
                    return [None]

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': '等待超时'
                    })
                    return [None]

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
                        'range': f"{start_idx + 1}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None]
            else:
                print(f"  Response looks normal; preparing to parse")
                break

        instruction = self.parse_uml_instruction(response)

        if instruction:
            is_valid, errors = self.validate_instruction_format(instruction)
            if is_valid:
                print(f"  Instruction format validation passed")
                return [instruction]
            else:
                print(f"  Instruction format validation failed: {errors}")
                return [instruction]
        else:
            print(f"  Parsing failed")
            return [None]

    def repair_file(self, csv_path):
        """Repair failed records in one file."""
        print(f"\n{'#' * 60}")
        print(f"# Processing file: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

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

        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        error_batches = self.detect_error_batches(df)

        if not error_batches:
            print("\n No erroneous data requiring repair found")
            self.print_error_report()
            return 0

        self.print_error_report()

        print(f"\n Found {len(error_batches)} erroneous batches, about {len(error_batches) * BATCH_SIZE} data items require repair")
        user_input = input("是否继续修复? (y/n): ").strip().lower()
        if user_input != 'y':
            print(" User cancelled repair")
            return 0

        self.start_new_chat()

        repaired_batches = 0
        for batch_num, start_idx, batch_data in error_batches:
            instructions = self.process_batch(batch_data, start_idx, batch_num)

            for i, instruction in enumerate(instructions):
                row_idx = start_idx + i
                if instruction:
                    instruction = self.normalize_three_part_format(instruction)
                    df.at[row_idx, 'Instruction'] = instruction
                    self.repaired_count += 1
                else:
                    df.at[row_idx, 'Instruction'] = "ERROR: 生成失败"

            repaired_batches += 1

            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  Progress saved: {start_idx + len(batch_data)}/{len(df)}")

            if repaired_batches < len(error_batches):
                self.start_new_chat()

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n File repair complete: {os.path.basename(csv_path)}")
        print(f"  Batches repaired: {repaired_batches}")
        print(f"  Data items repaired: {self.repaired_count} items\n")

        return self.repaired_count

    def run(self):
        """Run the workflow."""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'UML batch integrity repair system':^60}")
        print(f"{'=' * 60}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Batch size: {BATCH_SIZE} items/batch")
        period_check_status = "启用" if ENABLE_PERIOD_CHECK else "关闭"
        print(f"Checks: ERROR markers + three-part format + period check ({period_check_status})")
        print(f"Target file: {CSV_FILE}")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_repaired = self.repair_file(csv_path)
            else:
                print(f" File not found: {csv_path}")
                total_repaired = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'Repair complete':^60}")
            print(f"{'=' * 60}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed time: {duration}")
            print(f"Total repaired: {total_repaired} data items")

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
    repairer = UMLBatchRepairer()
    repairer.run()
