"""Repair failed image-domain instruction batches."""

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
DATASET_PATH = r"D:\MyPyProject\crowdsourcing_instruction_generator\dataset\image"
GPT_URL = "https://sass-node1.chatshare.biz/"

CSV_FILE = "image_interim_coco_1k.csv"

BATCH_SIZE = 1
CHECK_INTERVAL = 100
WAIT_NEW_RESPONSE_TIMEOUT = 60
CONTENT_STABLE_CHECKS = 3
ENABLE_PERIOD_CHECK = False

class ImageBatchRepairer:
    """Repair failed image-instruction batches."""
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
        print("正在初始化浏览器...")
        print("="*60)

        if not os.path.exists(CHROME_PATH):
            raise FileNotFoundError(f"Chrome浏览器路径不存在: {CHROME_PATH}")
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

    def clean_json_data(self, json_str):
        """Clean JSON data."""
        try:
            data = json.loads(json_str)

            cleaned = {
                "description": data.get("description", ""),
                "details": data.get("details", {})
            }

            return json.dumps(cleaned, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠ JSON清洗失败: {e}")
            return json_str

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

        raise NoSuchElementException("无法找到输入框")

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
            print(f"检测更新异常: {e}")
            return False

    def wait_for_response_complete(self, timeout=300):
        """Wait for the browser response to finish."""
        print("  等待生成...", end='', flush=True)
        start_time = time.time()
        last_progress_time = start_time

        print(" [等待响应]", end='', flush=True)
        response_appeared = False

        consecutive_validation_failures = 0
        MAX_VALIDATION_FAILURES = 10

        check_count = 0
        while time.time() - start_time < WAIT_NEW_RESPONSE_TIMEOUT:
            try:
                current_count = self.get_current_response_count()

                if current_count > self.response_count_before_send:
                    print(f" [检测到可能的新回复,验证中]", end='', flush=True)
                    time.sleep(2)

                    recheck_count = self.get_current_response_count()

                    if recheck_count > self.response_count_before_send:
                        if self._validate_new_response():
                            elapsed = int(time.time() - start_time)
                            response_appeared = True
                            print(f" ✓ [新回复已确认,耗时{elapsed}s]", end='', flush=True)
                            break
                        else:
                            consecutive_validation_failures += 1
                            print(f" [内容验证失败{consecutive_validation_failures}/{MAX_VALIDATION_FAILURES}]", end='',
                                  flush=True)

                            if consecutive_validation_failures >= MAX_VALIDATION_FAILURES:
                                elapsed = int(time.time() - start_time)
                                print(f" ⚠️ [验证失败但强制接受,耗时{elapsed}s]", end='', flush=True)
                                response_appeared = True
                                break

                            time.sleep(2)
                    else:
                        print(f" [数量未稳定,继续等待]", end='', flush=True)
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
            print(f" ✗ 等待响应超时({WAIT_NEW_RESPONSE_TIMEOUT}s)")
            return False

        time.sleep(1)
        print(" [检测完成]", end='', flush=True)

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
                        print(" ✓ 完成")
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
                print(f"⚠ [{str(e)[:20]}]", end='', flush=True)
                time.sleep(1)

        print(" ✓ 完成(达到检查上限)")
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
                    print(f"  ✓ 提取到回复 ({len(response_text)} 字符)")

                    has_definition = "Definition:" in response_text
                    has_emphasis = "Emphasis" in response_text or "Caution" in response_text
                    has_avoid = "Avoid" in response_text

                    if has_definition or has_emphasis or has_avoid:
                        print(f"  ✓ 内容验证通过（包含指令关键词）")
                    else:
                        print(f"  ⚠ 警告：回复可能不包含预期格式")

                    return response_text
                else:
                    print(f"  ⚠ 提取的内容太短: {len(response_text) if response_text else 0} 字符")

            print(f"  ✗ 无法提取有效回复（当前{current_count}条，发送前{self.response_count_before_send}条）")
            return ""

        except Exception as e:
            print(f"  ✗ 提取回复失败: {e}")
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
            print(f"[验证异常,接受]", end='', flush=True)
            return True

    def parse_instructions(self, response_text, expected_count):
        """Parse instructions."""
        instructions = []

        pattern = r'【图像\d+】\s*\n(.*?)(?=【图像\d+】|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if len(matches) == expected_count:
            for match in matches:
                instructions.append(match.strip())
        else:
            parts = response_text.split('Definition:')
            for part in parts[1:]:
                if 'Emphasis & Caution:' in part and 'Things to Avoid:' in part:
                    instructions.append('Definition:' + part.strip())

        return instructions

    def parse_image_instruction(self, response_text):
        """Parse image instruction."""
        response_text = response_text.strip()

        if 'Definition:' not in response_text or 'Emphasis & Caution:' not in response_text or 'Things to Avoid:' not in response_text:
            print(f"  ⚠ 缺少必要的标注")
            return None

        def_pos = response_text.find('Definition:')
        emp_pos = response_text.find('Emphasis & Caution:')
        avoid_pos = response_text.find('Things to Avoid:')

        if not (def_pos < emp_pos < avoid_pos):
            print(f"  ⚠ 标注顺序不正确")
            return None

        def_text = response_text[def_pos + len('Definition:'):emp_pos].strip()
        emp_text = response_text[emp_pos + len('Emphasis & Caution:'):avoid_pos].strip()
        avoid_text = response_text[avoid_pos + len('Things to Avoid:'):].strip()

        def_content = self._clean_and_merge_content(def_text)
        emp_content = self._clean_and_merge_content(emp_text)
        avoid_content = self._clean_and_merge_content(avoid_text)

        instruction = f"Definition: {def_content}\nEmphasis & Caution: {emp_content}\nThings to Avoid: {avoid_content}"
        return instruction

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

    def normalize_three_part_format(self, instruction):
        """Normalize three part format."""
        if not instruction:
            return instruction

        lines = instruction.strip().split('\n')
        if len(lines) >= 3:
            has_definition = any(line.strip().startswith('Definition:') for line in lines)
            has_emphasis = any(line.strip().startswith('Emphasis & Caution:') or line.strip().startswith('Emphasis and Caution:') for line in lines)
            has_avoid = any(line.strip().startswith('Things to Avoid:') for line in lines)

            if has_definition and has_emphasis and has_avoid:
                return instruction

        normalized_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if 'Definition:' in line:
                def_start = line.find('Definition:')
                def_content = line[def_start:]

                if 'Emphasis & Caution:' in def_content or 'Emphasis and Caution:' in def_content:
                    parts = re.split(r'(Emphasis & Caution:|Emphasis and Caution:|Things to Avoid:)', def_content)
                    for i, part in enumerate(parts):
                        if part.strip():
                            if i == 0:
                                normalized_lines.append(part.strip())
                            elif part.strip() in ['Emphasis & Caution:', 'Emphasis and Caution:', 'Things to Avoid:']:
                                continue
                            else:
                                if i > 0 and i < len(parts):
                                    prev = parts[i-1].strip()
                                    if prev in ['Emphasis & Caution:', 'Emphasis and Caution:']:
                                        normalized_lines.append(f"Emphasis & Caution: {part.strip()}")
                                    elif prev == 'Things to Avoid:':
                                        normalized_lines.append(f"Things to Avoid: {part.strip()}")
                else:
                    normalized_lines.append(def_content.strip())

            elif 'Emphasis & Caution:' in line or 'Emphasis and Caution:' in line:
                if 'Emphasis and Caution:' in line:
                    line = line.replace('Emphasis and Caution:', 'Emphasis & Caution:')
                normalized_lines.append(line)

            elif 'Things to Avoid:' in line:
                normalized_lines.append(line)

            else:
                continue

        if len(normalized_lines) >= 3:
            return '\n'.join(normalized_lines[:3])
        else:
            return instruction

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

    def process_batch(self, image_data_batch, start_idx):
        """Process batch."""
        print(f"\n{'=' * 60}")
        print(f"处理第 {start_idx + 1} 条图像数据")
        print(f"{'=' * 60}")

        header, description = image_data_batch[0]

        try:
            data = json.loads(description)
            filtered_data = {
                "description": data.get("description", ""),
                "details": data.get("details", {})
            }
            json_str = json.dumps(filtered_data, ensure_ascii=False, separators=(',', ':'))
        except json.JSONDecodeError:
            json_str = description

        prompt = f"""You are a computer vision data expert and crowdsourcing task designer. Based on the input image analysis structured data, write an English image annotation instruction for crowdsourcing workers.

Core Principles:
1. Annotation Focus: The instruction must explicitly require workers to draw bounding boxes.
2. Foreground Extraction: Extract main foreground objects (e.g., people, vehicles) from the objects list as annotation targets. Ignore background elements.
3. Direct Reference: Use English terms directly from the JSON data. Do not replace with synonyms.
4. Extreme Conciseness: Keep Emphasis and Avoid sections brief. Use "-" if no significant visual features or distractors exist.

Output Format Requirements:

Definition: Use a clear imperative sentence to describe the annotation targets. Must start with "In this task," and explicitly mention "draw bounding boxes around".
Emphasis & Caution: Only list highly distinctive visual features (e.g., specific colors, positions). Use "-" if nothing specific to emphasize.
Things to Avoid: Only list confusing background distractors. Use "-" if nothing specific to avoid.

CRITICAL RULES:
- Each section must be on a separate line
- Each line must start with the section label (Definition: / Emphasis & Caution: / Things to Avoid:)
- Definition must include "draw bounding boxes around" and list specific objects from JSON data
- Keep all sections concise
- Output ONLY these three lines, nothing else

Image analysis structured data (JSON format):
```json
{json_str}
```
"""

        max_retries = 3
        response = None
        retry_happened = False

        for retry_count in range(max_retries):
            if retry_count > 0:
                retry_happened = True
                print(f"\n🔄 检测到生成错误,正在重试 ({retry_count}/{max_retries - 1})...")
                print("  🔄 开启新对话以重试...")
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
                    return [None], retry_happened

            if not self.wait_for_response_complete():
                if retry_count < max_retries - 1:
                    continue
                else:
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': '等待超时'
                    })
                    return [None], retry_happened

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
                    print(f"  ✗ 已达到最大重试次数({max_retries}),放弃本条数据")
                    self.error_log.append({
                        'range': f"{start_idx + 1}",
                        'error': f'生成错误(重试{max_retries}次后失败)'
                    })
                    return [None], retry_happened
            else:
                print(f"  ✓ 响应正常,准备解析")
                break

        instruction = self.parse_image_instruction(response)

        if instruction:
            return [instruction], retry_happened
        else:
            print(f"  ✗ 解析失败")
            return [None], retry_happened

    def start_new_chat(self):
        """Start a new browser chat."""
        print("\n>>> 开启新对话...")
        try:
            from selenium.webdriver.common.action_chains import ActionChains

            print("  🔨 发送快捷键 Ctrl+Shift+O...")
            actions = ActionChains(self.driver)
            actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('o').key_up(Keys.SHIFT).key_up(
                Keys.CONTROL).perform()
            print("  ✓ 快捷键已发送")

            time.sleep(3)

            self.cached_input_selector = None
            self.cached_button_selector = None
            self.response_count_before_send = 0

            self.batches_since_refresh = 0

            print("  ✓ 新对话已就绪\n")

        except Exception as e:
            print(f"  ✗ 开启新对话失败: {e}")
            print("  ℹ 将继续在当前对话中处理")

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
                if 'bounding box' not in content.lower() and 'draw box' not in content.lower():
                    errors.append("Definition未明确要求画边框")
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
        print("开始检测错误数据...")
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

        print(f"\n检测结果:")
        print(f"  总数据条数: {total_rows}")
        print(f"  错误数据条数: {error_count}")
        print(f"  需修复批次数: {len(error_batches)}")

        return error_batches

    def print_error_report(self):
        """Print error report."""
        if not self.error_details:
            return

        print("\n" + "="*60)
        print("详细错误报告")
        print("="*60)

        for row_num, header, error_msg in self.error_details:
            print(f"\n第{row_num}行: {header}")
            print(f"  错误: {error_msg}")

        print("\n" + "="*60)

    def repair_file(self, csv_path):
        """Repair failed records in one file."""
        print(f"\n{'#' * 60}")
        print(f"# 处理文件: {os.path.basename(csv_path)}")
        print(f"{'#' * 60}")

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
                    raise Exception("无法读取文件")

        except Exception as e:
            print(f"✗ 读取文件失败: {e}")
            return 0

        if 'Instruction' not in df.columns:
            df['Instruction'] = ''
        df['Instruction'] = df['Instruction'].astype(str)
        df.loc[df['Instruction'] == 'nan', 'Instruction'] = ''

        error_batches = self.detect_error_batches(df)

        if not error_batches:
            print("\n✓ 未发现需要修复的错误数据")
            self.print_error_report()
            return 0

        self.print_error_report()

        print(f"\n⚠️ 发现 {len(error_batches)} 个错误批次,共约 {len(error_batches) * BATCH_SIZE} 条数据需要修复")
        user_input = input("是否继续修复? (y/n): ").strip().lower()
        if user_input != 'y':
            print("❌ 用户取消修复")
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
            print(f"  ✓ 已保存进度: {start_idx + len(batch_data)}/{len(df)}")

            if repaired_batches < len(error_batches):
                self.start_new_chat()

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件修复完成: {os.path.basename(csv_path)}")
        print(f"  修复批次: {repaired_batches}")
        print(f"  修复数据: {self.repaired_count} 条\n")

        return self.repaired_count

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
                    raise Exception("无法读取文件")

        except Exception as e:
            print(f"✗ 读取文件失败: {e}")
            return 0

        required_columns = ['Header', 'Description', 'Instruction']
        if not all(col in df.columns for col in required_columns):
            print(f"✗ CSV文件缺少必要的列: {required_columns}")
            print(f"  当前列: {df.columns.tolist()}")
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

        print(f"总计需处理: {total_rows} 条图像数据\n")
        print(f"刷新策略: 每{REFRESH_INTERVAL}条数据（{REFRESH_INTERVAL//BATCH_SIZE}批）或发生重试后刷新对话\n")

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
                print(f"\n  🔄 批次完成刷新 - 已处理{self.batches_since_refresh}批({self.batches_since_refresh * BATCH_SIZE}条数据)")
                print(f"  ℹ️ 当前进度: {i + BATCH_SIZE}/{total_rows}")
                self.start_new_chat()

            if (i + BATCH_SIZE) % 50 == 0:
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"\n  💾 已保存进度: {i + BATCH_SIZE}/{total_rows}\n")

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 文件处理完成: {os.path.basename(csv_path)}")
        print(f"  已处理 {total_rows} 条图像数据\n")

        return total_rows

    def run(self):
        """Run the workflow."""
        start_time = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"{'Image批次完整性修复系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        period_check_status = "启用" if ENABLE_PERIOD_CHECK else "关闭"
        print(f"检测功能: ERROR标记 + 三段式格式 + 句号检查({period_check_status})")
        print(f"目标文件: {CSV_FILE}")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_repaired = self.repair_file(csv_path)
            else:
                print(f"✗ 文件不存在: {csv_path}")
                total_repaired = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'修复完成':^60}")
            print(f"{'=' * 60}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计修复: {total_repaired} 条数据")

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
    repairer = ImageBatchRepairer()
    repairer.run()
