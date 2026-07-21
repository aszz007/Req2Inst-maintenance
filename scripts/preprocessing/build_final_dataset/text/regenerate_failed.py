"""Repair failed text-domain instruction batches."""

import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import re
from datetime import datetime
import chardet

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"D:\MyPyProject\crowdsourcing_instruction_generator\dataset\Requirements_data\Text_data"
GPT_URL = "https://sass-node1.chatshare.biz/"

CSV_FILE = "enhanced_CCHIT.csv"

BATCH_SIZE = 10
CHECK_INTERVAL = 100
WAIT_NEW_RESPONSE_TIMEOUT = 60
CONTENT_STABLE_CHECKS = 3
ENABLE_PERIOD_CHECK = False

class TextBatchRepairer:
    """Repair failed text-instruction batches."""
    def __init__(self):
        self.driver = None
        self.current_tab = None
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
        print(" Chrome path validated successfully")

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
            print(" ChromeDriver started successfully")

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
        if debug:
            print(" Locating input field...")

        if self.cached_input_selector:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.cached_input_selector))
                )
                if element.is_displayed() and element.is_enabled():
                    if debug:
                        print("  Cached selector used successfully")
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
                                parent = elem.find_element(By.XPATH, "./ancestor::article[contains(@data-testid, 'conversation-turn')]")
                                if parent:
                                    turn_attr = parent.get_attribute("data-turn")
                                    if turn_attr == "assistant":
                                        valid_count += 1
                            except:
                                continue
                        if valid_count > 0:
                            return valid_count
                    else:
                        if len(elements) > 0:
                            return len(elements)

                except:
                    continue

            print("  No assistant response found; returning 0")
            return 0

        except Exception as e:
            print(f"  Unexpected response count: {e}")
            return 0

    def wait_for_response(self, timeout=60):
        """Wait for a browser response."""
        print("\n Waiting for response generation...")
        start_time = time.time()

        target_count = self.response_count_before_send + 1
        print(f"  Expected response count: {target_count}")

        last_content = ""
        stable_count = 0

        while time.time() - start_time < timeout:
            try:
                current_count = self.get_current_response_count()

                if current_count >= target_count:
                    print(f"  New response detected (current {current_count} items)")

                    response_selectors = [
                        "div[data-message-author-role='assistant']",
                        "article[data-turn='assistant']",
                    ]

                    for selector in response_selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if len(elements) >= target_count:
                                response_elem = elements[target_count - 1]
                                current_content = response_elem.text

                                if current_content == last_content:
                                    stable_count += 1
                                    print(f"  Content stable ({stable_count}/{CONTENT_STABLE_CHECKS})", end='\r', flush=True)

                                    if stable_count >= CONTENT_STABLE_CHECKS:
                                        print("\n Content is stable")
                                        return current_content
                                else:
                                    last_content = current_content
                                    stable_count = 0

                                time.sleep(1)
                                break
                        except:
                            continue

                time.sleep(2)

            except Exception as e:
                print(f"  Check error: {e}")
                time.sleep(2)

        print(f"\n Response wait timed out ({timeout} seconds)")
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

    def parse_text_instruction(self, response_text):
        """Parse text instruction."""
        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
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

    def send_prompt(self, prompt_text, max_retries=3):
        """Submit a prompt to the browser session."""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print("\n Sending prompt...")
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
                    print("  Clicked the Send button")
                else:
                    input_box.send_keys(Keys.RETURN)
                    print("  Sent with Enter")

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

    def process_batch(self, batch_data, start_idx):
        """Process batch."""
        print(f"\n{'=' * 60}")
        print(f"Processing batch {start_idx + 1}-{start_idx + len(batch_data)}")
        print(f"{'=' * 60}")

        requirements_text = ""
        for i, (idx, requirement) in enumerate(batch_data):
            requirements_text += f"【需求{i+1}】\n{requirement}\n\n"

        prompt = f"""你是一个众包任务设计专家。请根据以下输入的需求文本,编写一个适合众包工人使用的英文任务指令。

核心原则:
1.极致精简:众包工人时间宝贵,请使用最简练的语言。
2.结构规范:严格按照下方定义的格式输出。
3.英语输出:无论输入是何种语言,输出必须是英文。

格式要求:
-Definition:使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
-Emphasis & Caution:仅指出极易出错或必须满足的特定条件。如无特别强调,填入 "-"。
-Things to Avoid:仅列出禁止的操作。如无特别避免事项,填入 "-"。

请为以下{len(batch_data)}条需求分别生成指令,严格按照以下格式输出:

{requirements_text}

输出格式:
【需求1】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

【需求2】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ..."""

        retry_happened = False
        max_retries = 3

        for retry in range(max_retries):
            if retry > 0:
                print(f"\n Batch retry {retry}/{max_retries-1}")
                retry_happened = True

            if not self.send_prompt(prompt):
                print("  Failed to send prompt")
                if retry < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(batch_data)}",
                        'error': f'提示词发送失败(重试{max_retries}次后失败)'
                    })
                    return [None] * len(batch_data), retry_happened

            response = self.wait_for_response(timeout=WAIT_NEW_RESPONSE_TIMEOUT)

            if not response or len(response) < 50:
                print("  Response is empty or too short")
                if retry < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(batch_data)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(batch_data), retry_happened
            else:
                print("  Response looks normal; preparing to parse")
                break

        instructions = []
        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        if len(matches) == len(batch_data):
            for match in matches:
                instruction = match.strip()
                instruction = self.normalize_three_part_format(instruction)
                instructions.append(instruction)
        else:
            parts = response.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instruction = 'Definition:' + part.strip()
                    instruction = self.normalize_three_part_format(instruction)
                    instructions.append(instruction)

        if len(instructions) != len(batch_data):
            print(f"  Parsed count mismatch: expected {len(batch_data)}, actual {len(instructions)}")
            return [None] * len(batch_data), retry_happened

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

            print("  New conversation ready\n")

        except Exception as e:
            print(f"  Failed to open a new conversation: {e}")
            print("  Continuing in the current conversation")

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
                requirement_id = df.loc[idx, 'Requirement ID']
                row_num = idx + 1

                if 'ERROR' in instruction.upper() and '生成失败' in instruction:
                    batch_has_error = True
                    error_msg = "含有ERROR标记"
                    batch_error_details.append((row_num, requirement_id, error_msg))
                    continue

                if not instruction or instruction.strip() == '' or instruction == 'nan':
                    batch_has_error = True
                    error_msg = "指令为空"
                    batch_error_details.append((row_num, requirement_id, error_msg))
                    continue

                is_valid, format_errors = self.validate_instruction_format(instruction)
                if not is_valid:
                    batch_has_error = True
                    error_msg = "; ".join(format_errors)
                    batch_error_details.append((row_num, requirement_id, error_msg))

            if batch_has_error:
                error_count += len(batch_error_details)
                self.error_details.extend(batch_error_details)

                batch_data = []
                for idx in range(i, batch_end):
                    requirement_id = df.loc[idx, 'Requirement ID']
                    requirement_text = df.loc[idx, 'Requirement']
                    batch_data.append((requirement_id, requirement_text))

                error_batches.append((len(error_batches) + 1, i, batch_data))

        print("\nScan results:")
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

        for row_num, req_id, error_msg in self.error_details:
            print(f"\nRow {row_num} line: {req_id}")
            print(f"  Error: {error_msg}")

        print("\n" + "="*60)

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
            instructions, _ = self.process_batch(batch_data, start_idx)

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
        print(f"{'Text batch integrity repair system':^60}")
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
                print("\nError log:")
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
    repairer = TextBatchRepairer()
    repairer.run()
