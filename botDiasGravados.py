from fuzzywuzzy import process
from PIL import ImageGrab
import pandas as pd
import numpy as np
import pytesseract
import pyautogui
import time
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:/Program Files/Tesseract-OCR/tessdata/'

def execute_command(command):
    action, *params = command
    if action == "click":
        x, y = params
        pyautogui.click(x, y)
    elif action == "doubleclick":
        x, y = params
        pyautogui.click(x, y, clicks=2)
    elif action == "move":
        x, y = params
        pyautogui.moveTo(x, y)
    elif action == "write":
        text = params[0]
        pyautogui.write(text)
    elif action == "press":
        key = params[0]
        pyautogui.press(key)
    elif action == "hotkey":
        keys = params
        pyautogui.hotkey(*keys)
    elif action == "count_green":
        x, y, width, height = params
        return count_green_items(x, y, width, height)
    elif action == "count_template":
        x, y, width, height, template_path = params
        return count_template_items(x, y, width, height, template_path)
    elif action == "find_text":
        x, y, width, height, search_texts = params
        for search_text in search_texts:
            found, location = find_text(x, y, width, height, search_text)
            if found and location:
                pyautogui.moveTo(location[0], location[1])
                pyautogui.click(location[0], location[1], clicks=2)
                return found
        return False
    elif action == "wait_for_loading":
        x, y, width, height, template_path = params
        wait_for_loading_to_disappear(x, y, width, height, template_path)
    elif action == "pause":
        pause_time = params[0]
        time.sleep(pause_time)
    else:
        print(f"Comando '{action}' não reconhecido.")
        return None

def correct_text(text):
    corrections = {
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def find_text(x, y, width, height, search_texts):
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot_np = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    data = pytesseract.image_to_data(screenshot_gray, lang='eng', output_type=pytesseract.Output.DICT)
    recognized_texts = [correct_text(word).upper() for word in data['text']]
    best_match, best_score = process.extractOne(search_texts, recognized_texts)
    
    if best_score > 60:
        index = recognized_texts.index(best_match)
        x_offset = data['left'][index]
        y_offset = data['top'][index]
        location = (x + x_offset, y + y_offset)
        return True, location
    return False, None


def count_green_items(x, y, width, height):
    time.sleep(2)
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    lower_green = np.array([40, 100, 50])
    upper_green = np.array([80, 255, 255])

    hsv = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def count_template_items(x, y, width, height, template_path):
    time.sleep(2)
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    screenshot_gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)

    count = 0
    for pt in zip(*loc[::-1]):
        count += 1
    return count

def wait_for_loading_to_disappear(x, y, width, height, template_path, timeout=60):
    start_time = time.time()
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    while True:
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val < 0.8:
            break

        if time.time() - start_time > timeout:
            break

        time.sleep(1) 

cameras = []

commands_template = [
    ("move", 340, 118),
    ("click", 340, 118),
    ("write", ""),
    ("press", "enter"),
    ("find_text", 290, 200, 320, 250, cameras), 
    ("move", 1257, 1060),
    ("wait_for_loading", 617, 175, 1200, 600, "C:/Users/Suporte/Desktop/Bots/loading.png"), 
    ("pause", 5),
    ("click", 1257, 1060),
    ("move", 1200, 980),
    ("click", 1200, 980),
    ("move", 1159, 760),
    ("count_green", 1150, 755, 235, 280),
    ("click", 1159, 760),
    ("count_green", 1150, 755, 235, 280),
    ("count_template", 1150, 755, 235, 280, "C:/Users/Suporte/Desktop/Bots/imagemVermelha.png"),
    ("click", 1159, 760),
    ("count_green", 1150, 755, 235, 280),
    ("move", 1809, 56),
    ("doubleclick", 1809, 56),
    ("move", 340, 118),
    ("click", 340, 118),
    ("hotkey", "ctrl", "a"),
    ("press", "backspace")
]

def carregar_cameras_de_excel(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo, header=None)
    cameras = df[0].tolist()
    return cameras

def main():
    print("Você tem 5 segundos para alternar para o aplicativo...")
    time.sleep(5)

    caminho_arquivo_excel = r'C:/Users/Suporte/Desktop/Bots/Cameras.xlsx'
    cameras = carregar_cameras_de_excel(caminho_arquivo_excel)
    
    results = {}
    not_found_cameras = []

    for camera in cameras:
        total_count = 0
        commands = [(cmd if cmd[0] != "find_text" else ("find_text", cmd[1], cmd[2], cmd[3], cmd[4], cameras)) for cmd in commands_template]
        commands = [(cmd if cmd[0] != "write" else ("write", camera)) for cmd in commands]

        camera_found = False 

        for command in commands:
            result = execute_command(command)
            if command[0] == "find_text":
                if result:
                    camera_found = True
                else:
                    print(f"Nenhum texto correspondente encontrado para {camera}. Pulando para a próxima câmera.")
                    not_found_cameras.append(camera)
                    additional_commands = [
                        ("move", 340, 118),
                        ("hotkey", "ctrl", "a"),
                        ("press", "backspace"),
                    ]
                    for additional_command in additional_commands:
                        execute_command(additional_command)
                    break  
            elif result is not None and (command[0] == "count_green" or command[0] == "count_template"):
                total_count += result
            time.sleep(1)

        if camera_found:  
            results[camera] = total_count

    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Dias Gravados'])
    df_results['Dias Faltando'] = 60 - df_results['Dias Gravados']
    df_results.index.name = 'Câmeras'

    if not_found_cameras:
        df_not_found = pd.DataFrame(not_found_cameras, columns=['Câmeras Não Encontradas'])

    file_path_results = os.path.join(os.environ['USERPROFILE'], 'Desktop', "ContagemDias.xlsx")
    file_path_not_found = os.path.join(os.environ['USERPROFILE'], 'Desktop', "CamerasNaoEncontradas.xlsx")

    df_results.to_excel(file_path_results, index=True)

    if not_found_cameras:
        df_not_found.to_excel(file_path_not_found, index=False)

    print(f"Resultados das câmeras encontradas salvos em {file_path_results}")
    if not_found_cameras:
        print(f"Lista de câmeras não encontradas salva em {file_path_not_found}")

if __name__ == "__main__":
    main()