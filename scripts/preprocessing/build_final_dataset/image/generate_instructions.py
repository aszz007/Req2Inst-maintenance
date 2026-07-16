"""Generate image-domain crowdsourcing instructions with browser-assisted batching."""

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
REFRESH_INTERVAL = 1
CHECK_INTERVAL = 100
TEST_MODE_LIMIT = 10

WAIT_NEW_RESPONSE_TIMEOUT = 60
CONTENT_STABLE_CHECKS = 3

IMAGE_ANNOTATION_EXAMPLE = {
    "json": """{
  "description": "A busy urban street with cars and traffic signs in the foreground.",
  "details": {
    "objects": ["car", "traffic sign", "person", "building", "road"],
    "scene": "urban street",
    "spatial_info": "Cars are in the foreground, buildings in the background"
  }
}""",
    "instruction": """Definition: In this task, draw bounding boxes around all "car", "traffic sign", and "person" objects.
Emphasis & Caution: Focus on foreground objects. Ensure bounding boxes are tight and accurate.
Things to Avoid: Do not annotate "building" or "road" as these are background elements."""
}

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
        print(f"{'批量图像标注指令生成系统':^60}")
        print(f"{'=' * 60}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模式: {'测试模式 (10条)' if self.test_mode else '完整模式'}")
        print(f"批次大小: {BATCH_SIZE} 条/批")
        print(f"刷新间隔: {REFRESH_INTERVAL} 条 ({REFRESH_INTERVAL//BATCH_SIZE} 批)")
        print(f"响应超时: {WAIT_NEW_RESPONSE_TIMEOUT} 秒")
        print(f"{'=' * 60}\n")

        try:
            self.init_driver()

            csv_path = os.path.join(DATASET_PATH, CSV_FILE)
            if os.path.exists(csv_path):
                total_processed = self.process_file(csv_path)
            else:
                print(f"✗ 文件不存在: {csv_path}")
                total_processed = 0

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print(f"{'处理完成':^60}")
            print(f"{'=' * 60}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"耗时: {duration}")
            print(f"总计处理: {total_processed} 条图像数据")
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
    print(f"  1. 测试模式 (仅处理前{TEST_MODE_LIMIT}条)")
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
