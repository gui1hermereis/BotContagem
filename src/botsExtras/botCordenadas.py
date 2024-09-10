import pyautogui
import time

def show_coordinates():
    print("Movimente o mouse para a posição desejada e pressione Ctrl+C para parar.")
    try:
        while True:
            x, y = pyautogui.position() 
            print(f"Coordenadas: ({x}, {y})", end='\r') 
            time.sleep(0.1)  
    except KeyboardInterrupt:
        print("\nCaptura de coordenadas encerrada.")

if __name__ == "__main__":
    show_coordinates()
