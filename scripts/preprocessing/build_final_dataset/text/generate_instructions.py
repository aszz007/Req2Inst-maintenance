"""Generate text-domain crowdsourcing instructions with browser-assisted batching."""

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

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATASET_PATH = r"D:\MyPyProject\crowdsourcing_instruction_generator\dataset\Requirements_data\Text_data"
GPT_URL = "https://sass-node1.chatshare.biz/"

CSV_FILES = [
    "enhanced_CCHIT.csv",
    "enhanced_CM1.csv",
    "enhanced_GANNT.csv",
    "enhanced_InfusionPump.csv",
    "enhanced_Modis.csv",
    "enhanced_WARC.csv"
]

BATCH_SIZE = 10
REFRESH_INTERVAL = 50
CHECK_INTERVAL = 100
TEST_MODE_LIMIT = 50

SYSTEM_PROMPT = """你是一个众包任务设计专家。请根据以下输入的需求文本,编写一个适合众包工人使用的英文任务指令。

核心原则:
1.极致精简:众包工人时间宝贵,请使用最简练的语言。
2.结构规范:严格按照下方定义的格式输出。
3.英语输出无论输入是何种语言,输出必须是英文。

格式要求:
-Definition:使用简明扼要的祈使句描述主要目标。必须以 "In this task," 开头。
-Emphasis & Caution:仅指出极易出错或必须满足的特定条件。如无特别强调,填入 "-"。
-Things to Avoid:仅列出禁止的操作。如无特别避免事项,填入 "-"。

请为以下{count}条需求分别生成指令,严格按照以下格式输出:

{requirements}

请严格按照以下格式输出每条指令,不要添加额外说明:

【需求1】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

【需求2】
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...

(依此类推)
"""

QUALITY_CHECK_PROMPT = """请检查以下需求和生成的指令是否对应正确。如果有问题,请指出哪些需求的指令不匹配或质量不佳。

{check_content}

请逐条评估并指出问题,如果全部正确请回复"全部正确"。"""


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

    def init_driver(self):
        """Initialize the browser driver."""
        print("\n" + "="*60)
        print("正在初始化浏览器...")
        print("="*60)

        if not os.path.exists(CHROME_PATH):
            raise FileNotFoundError(f"Chrome executable not found at: {CHROME_PATH}")
        print(f"✓ Chrome路径验证成功")

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

            print("✓ ChromeOptions配置完成")
            print("正在启动ChromeDriver...")
            self.driver = webdriver.Chrome(options=options)
            print(f"✓ ChromeDriver启动成功")

            print(f"\n正在导航到: {GPT_URL}")
            self.driver.get(GPT_URL)
            time.sleep(8)

            print(f"✓ 页面加载完成: {self.driver.title}")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n✗ 浏览器初始化失败: {e}")
            raise

    def find_input_box(self, debug=False):
        """Find input box."""
        if debug:
            print("🔍 定位输入框...")

        if self.cached_input_selector:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.cached_input_selector))
                )
                if element.is_displayed() and element.is_enabled():
                    if debug:
                        print(f"  ✓ 使用缓存选择器成功")
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
                    print(f"  尝试: {selector}")

                element = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )

                if element.is_displayed() and element.is_enabled():
                    self.cached_input_selector = selector
                    if debug:
                        print(f"  ✓ 成功: {selector}")
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
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
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
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    current_count = len(elements)

                    if current_count <= self.response_count_before_send:
                        return True

                    if elements and current_count > 0:
                        new_response = elements[-1]
                        first_text = new_response.text
                        first_len = len(first_text)

                        time.sleep(0.8)

                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if len(elements) > 0:
                            new_response = elements[-1]
                            second_text = new_response.text
                            second_len = len(second_text)

                            if second_len > first_len:
                                return True
                            return False
                except:
                    continue

            return False
        except:
            return False

    def wait_for_response_complete(self, timeout=300):
        """Wait for the browser response to finish."""
        print("  等待生成...", end='', flush=True)
        start_time = time.time()
        last_dot_time = start_time

        print(" [等待响应]", end='', flush=True)
        response_appeared = False

        for _ in range(20):
            try:
                current_count = self.get_current_response_count()
                if current_count > self.response_count_before_send:
                    response_appeared = True
                    print(" ✓", end='', flush=True)
                    break
            except:
                pass
            time.sleep(0.5)

        if not response_appeared:
            print(" [未检测到新响应，继续等待]", end='', flush=True)

        stable_count = 0
        required_stable_checks = 3

        while time.time() - start_time < timeout:
            try:
                is_updating = self.check_response_still_updating()

                if is_updating:
                    stable_count = 0
                    print(".", end='', flush=True)
                else:
                    stable_count += 1

                    if stable_count >= required_stable_checks:
                        print(" ✓ 完成")
                        return True
                    else:
                        print(".", end='', flush=True)

                current_time = time.time()
                if current_time - last_dot_time >= 5:
                    elapsed = int(current_time - start_time)
                    print(f" [{elapsed}s]", end='', flush=True)
                    last_dot_time = current_time

                time.sleep(0.5)

            except Exception as e:
                print(f" ⚠", end='', flush=True)
                time.sleep(1)

        print(" ✗ 超时")
        return False

    def extract_response(self):
        """Extract response."""
        try:
            response_selectors = [
                "div[class*='markdown']",
                "div[data-message-author-role='assistant']",
                "div[class*='message']",
                "[class*='assistant']"
            ]

            for selector in response_selectors:
                try:
                    response_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    current_count = len(response_elements)

                    if current_count > self.response_count_before_send:
                        last_response = response_elements[-1].text
                        if last_response and len(last_response) > 10:
                            print(f"  ✓ 提取到回复 ({len(last_response)} 字符)")
                            return last_response
                except:
                    continue

            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            print(f"  ⚠ 使用body文本")
            return body_text

        except Exception as e:
            print(f"✗ 提取回复失败: {e}")
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

    def parse_instructions(self, response_text, expected_count):
        """Parse instructions."""
        instructions = []

        pattern = r'【需求\d+】\s*\n(.*?)(?=【需求\d+】|$)'
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
                    print(f"\n📤 发送提示词...")
                    self.response_count_before_send = self.get_current_response_count()
                    print(f"  📊 当前页面已有 {self.response_count_before_send} 条回复")
                else:
                    print(f"  🔄 重试 {attempt}/{max_retries-1}...")

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

                print(f"  ✓ 文本设置成功 ({len(current_value)} 字符)")

                button = self.find_submit_button()
                if button:
                    self.driver.execute_script("arguments[0].click();", button)
                    print(f"  ✓ 点击发送按钮")
                else:
                    input_box.send_keys(Keys.RETURN)
                    print(f"  ✓ 使用Enter发送")

                time.sleep(2)

                check_value = input_box.get_attribute("value") if tag_name == "textarea" else input_box.text
                if not check_value or len(check_value.strip()) < 50:
                    print("  ✓ 确认消息已发送")
                    return True

                time.sleep(2)
                return True

            except Exception as e:
                print(f"  ✗ 发送异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return False

        return False

    def process_batch(self, requirements_batch, start_idx):
        """Process batch."""
        print(f"\n{'=' * 60}")
        print(f"处理第 {start_idx + 1}-{start_idx + len(requirements_batch)} 条需求")
        print(f"{'=' * 60}")

        req_text = ""
        for i, req in enumerate(requirements_batch, 1):
            req_text += f"{i}. {req}\n\n"

        prompt = SYSTEM_PROMPT.format(
            count=len(requirements_batch),
            requirements=req_text
        )

        max_retries = 3
        response = None

        for retry_count in range(max_retries):
            if retry_count > 0:
                print(f"\n🔄 检测到生成错误,正在重试 ({retry_count}/{max_retries - 1})...")
                time.sleep(3)

            if not self.send_prompt(prompt):
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': '发送失败'
                    })
                    return [None] * len(requirements_batch)

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': '等待超时'
                    })
                    return [None] * len(requirements_batch)

            response = self.extract_response()
            print(f"\n响应预览: {response[:200]}...\n")

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
                print(f"  ⚠️ 检测到生成错误: {response[:100]}")
                if retry_count < max_retries - 1:
                    print(f"  ↻ 将在3秒后重新发送...")
                    continue
                else:
                    print(f"  ✗ 已达到最大重试次数({max_retries}),放弃本批次")
                    self.error_log.append({
                        'range': f"{start_idx + 1}-{start_idx + len(requirements_batch)}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None] * len(requirements_batch)
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        instructions = self.parse_instructions(response, len(requirements_batch))

        if len(instructions) != len(requirements_batch):
            print(f"  ⚠ 警告: 期望{len(requirements_batch)}条,实际{len(instructions)}条")
            while len(instructions) < len(requirements_batch):
                instructions.append(None)
            instructions = instructions[:len(requirements_batch)]

        return instructions

    def start_new_chat(self):
        """Start a new browser chat."""
        print("\n>>> 开启新对话...")
        try:
            from selenium.webdriver.common.action_chains import ActionChains

            print("  📨 发送快捷键 Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  ✓ 快捷键已发送")

            time.sleep(3)

            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            print("  ✓ 新对话已就绪")

        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            print("  ℹ 将继续在当前对话中处理")

    def process_file(self, csv_path):
        """Process file."""
        print(f"\n{'#' * 60}")
        print(f"# 处理文件: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

        self.start_new_chat()

        try:
            with open(csv_path, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"文件编码: {encoding}")

            try:
                df = pd.read_csv(csv_path, encoding=encoding)
            except:
                for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc)
                        print(f"  ✓ 使用 {enc} 编码")
                        break
                    except:
                        continue
                else:
                    raise Exception("Failed to read the file")

        except Exception as e:
            print(f"✗ 读取文件失败: {e}")
            return 0

        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        total_rows = len(df)

        if self.test_mode:
            limit = min(TEST_MODE_LIMIT, total_rows)
            print(f"*** 测试模式: 仅处理前 {limit} 条 ***\n")
            df = df.head(limit)
            total_rows = limit

        print(f"总计需处理: {total_rows} 条需求\n")

        file_processed_count = 0

        for i in range(0, total_rows, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_rows)
            batch_requirements = df.loc[i:batch_end - 1, 'Low_Requirements'].tolist()

            instructions = self.process_batch(batch_requirements, i)

            for j, instruction in enumerate(instructions):
                if instruction:
                    df.at[i + j, 'Instruction'] = instruction
                else:
                    df.at[i + j, 'Instruction'] = "ERROR: 生成失败"

            self.processed_count += len(instructions)
            file_processed_count += len(instructions)

            if (file_processed_count % REFRESH_INTERVAL == 0 and
                    file_processed_count < total_rows):
                self.start_new_chat()

            if (i + BATCH_SIZE) % 50 == 0:
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"  ✓ 已保存进度: {i + BATCH_SIZE}/{total_rows}")

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件处理完成: {os.path.basename(csv_path)}")
        print(f"  已处理 {total_rows} 条需求\n")

        return total_rows

    def run(self):
        """Run the workflow."""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'批量指令生成系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模式: {'测试模式 (50条)' if self.test_mode else '完整模式'}")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            total_processed = 0
            for csv_file in CSV_FILES:
                csv_path = os.path.join(DATASET_PATH, csv_file)
                if os.path.exists(csv_path):
                    processed = self.process_file(csv_path)
                    total_processed += processed
                else:
                    print(f"✗ 文件不存在: {csv_file}")

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'处理完成':^60}")
            print(f"{'=' * 60}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计处理: {total_processed} 条需求")
            print(f"成功: {total_processed - len(self.error_log)} 条")
            print(f"失败: {len(self.error_log)} 条")

            if self.error_log:
                print(f"\n错误日志:")
                for error in self.error_log:
                    print(f"  - {error['range']}: {error['error']}")

            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"\n✗ 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.driver:
                input("按 Enter 关闭浏览器...")
                self.driver.quit()
                print("✓ 浏览器已关闭")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("请选择运行模式:")
    print("  1. 测试模式 (每个文件仅处理前50条)")
    print("  2. 完整模式 (处理所有数据)")
    print("="*60)

    while True:
        choice = input("\n请输入选项 (1 或 2): ").strip()
        if choice == "1":
            test_mode = True
            print("\n✓ 已选择: 测试模式")
            break
        elif choice == "2":
            test_mode = False
            print("\n✓ 已选择: 完整模式")
            confirm = input("⚠ 完整模式将处理大量数据,确认继续? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                print("已取消")
                exit(0)
        else:
            print("✗ 无效选项,请输入 1 或 2")

    automator = GPTAutomator(test_mode=test_mode)
    automator.run()
