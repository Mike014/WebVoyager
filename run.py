import argparse
import time
import json
import os
import logging
import base64
from datetime import datetime
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from smolagents import Tool, CodeAgent, HfApiModel

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("API Key for Hugging Face not found! Check the .env file.")

# Generate timestamp for result directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H_%M_%S")
RESULTS_DIR = os.path.join("results", TIMESTAMP)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logger configuration
LOG_FILE_PATH = os.path.join(RESULTS_DIR, 'run.log')

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding="utf-8"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# WebDriver configuration
def driver_config(args):
    options = webdriver.ChromeOptions()
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-running-insecure-content")
    if args.headless:
        options.add_argument("--headless")

    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
    options.add_experimental_option("prefs", {
        "download.default_directory": args.download_dir,
        "plugins.always_open_pdf_externally": True
    })

    service = Service(ChromeDriverManager().install())
    return options, service

# Function to encode an image in Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Tool for browser actions
class BrowserActionTool(Tool):
    """
    This tool performs browser actions like clicking, typing, scrolling, and extracting content.
    """
    name = "browser_action"
    description = "Performs actions such as visit, click, text input, scrolling, and extracting content."

    inputs = {
        "action": {"type": "string", "description": "Action to perform (visit, click, input, scroll, extract)."},
        "target": {"type": "string", "description": "CSS Selector of the target element.", "nullable": True},
        "value": {"type": "string", "description": "Value to input (if applicable).", "nullable": True},
    }
    output_type = "string"

    def __init__(self, driver):
        """Initialize with a WebDriver instance"""
        self.driver = driver
        self.is_initialized = True  # Fix per l'errore di inizializzazione

    def forward(self, action, target=None, value=None):
        """
        Executes browser actions.
        """
        try:
            if action == "visit":
                self.driver.get(target)
                time.sleep(3)
                logging.info(f"Visited: {target}")
                return f"Visited: {target}"

            elif action == "click":
                element = self.driver.find_element(By.CSS_SELECTOR, target)
                element.click()
                logging.info(f"Clicked on: {target}")
                return f"Clicked on: {target}"

            elif action == "input" and value:
                element = self.driver.find_element(By.CSS_SELECTOR, target)
                element.clear()
                element.send_keys(value)
                logging.info(f"Entered '{value}' in {target}")
                return f"Entered '{value}' in {target}"

            elif action == "scroll":
                element = self.driver.find_element(By.CSS_SELECTOR, target)
                self.driver.execute_script("arguments[0].scrollIntoView();", element)
                logging.info(f"Scrolled to: {target}")
                return f"Scrolled to: {target}"

            elif action == "extract":
                element = self.driver.find_element(By.CSS_SELECTOR, target)
                text = element.text
                logging.info(f"Extracted content from {target}: {text}")
                return text

            return f"Unknown action: {action}"
        
        except Exception as e:
            logging.error(f"Error executing '{action}' on {target}: {e}")
            return f"Error executing '{action}' on {target}: {e}"

# Create SmolAgents agent with WebDriver
def create_agent(driver):
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")
    return CodeAgent(
        tools=[BrowserActionTool(driver)], 
        model=model, 
        additional_authorized_imports=["requests", "bs4"]
    )

# Execute a task with SmolAgents
def process_task(task, driver_task):
    logging.info(f"Processing TASK {task['id']} - {task['web']}")

    driver_task.get(task['web'])
    time.sleep(3)

    # Capture screenshot
    img_path = os.path.join(RESULTS_DIR, 'screenshot.png')
    driver_task.save_screenshot(img_path)
    b64_img = encode_image(img_path)

    # Get page title
    page_title = driver_task.title
    logging.info(f"Loaded page: {page_title}")

    # Analyze the page with SmolAgents
    agent = create_agent(driver_task)
    prompt = f"Analyze the page {task['web']} and list all real interactive elements."
    response = agent.run(prompt)

    # Convert response to string if it's a list
    if isinstance(response, list):
        response = "\n".join(response)

    logging.info(f"Agent response:\n{response}")

    # Save response to file
    response_file_path = os.path.join(RESULTS_DIR, 'response.txt')
    with open(response_file_path, 'w', encoding='utf-8') as f:
        f.write(response)

    return response

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/tasks_test.jsonl')
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--download_dir", type=str, default="downloads")

    args = parser.parse_args()
    options, service = driver_config(args)
    driver_task = webdriver.Chrome(service=service, options=options)

    # Load tasks from JSONL file
    with open(args.test_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line.strip()) for line in f if line.strip()]

    for task in tasks:
        process_task(task, driver_task)

    driver_task.quit()
    logging.info("Process completed successfully!")

if __name__ == '__main__':
    main()
