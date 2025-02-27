import cv2
import numpy as np
from pyzbar.pyzbar import decode
import requests
import time

def get_camera_url():
    # IP адрес ESP32-CAM (замените на ваш)
    ESP32_CAM_IP = "192.168.31.170"
    # Меняем порт на 80 и путь на /capture для получения отдельных кадров
    return f"http://{ESP32_CAM_IP}/capture"

def process_qr(frame):
    # Декодирование QR-кода
    decoded_objects = decode(frame)
    
    for obj in decoded_objects:
        # Получение координат QR-кода
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        
        # Преобразование точек в numpy массив
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Рисование контура
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Вычисление центра QR-кода
        center_x = sum(point[0][0] for point in points) // len(points)
        center_y = sum(point[0][1] for point in points) // len(points)
        
        # Рисование красной точки в центре
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Вывод координат
        text = f"X: {center_x}, Y: {center_y}"
        cv2.putText(frame, text, (center_x - 50, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Вывод данных QR-кода
        data = obj.data.decode('utf-8')
        print(f"Данные QR-кода: {data}")
        print(f"Координаты центра: ({center_x}, {center_y})")
        
        return center_x, center_y, data
    
    return None, None, None

def main():
    url = get_camera_url()
    print(f"Подключение к камере по адресу: {url}")
    
    while True:
        try:
            # Получаем изображение напрямую через requests
            response = requests.get(url)
            if response.status_code == 200:
                # Преобразуем изображение из байтов в массив numpy
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Обработка QR-кода
                    x, y, data = process_qr(frame)
                    
                    # Отображение кадра
                    cv2.imshow("QR Scanner", frame)
            
            # Задержка между кадрами
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Ошибка при получении кадра: {e}")
            time.sleep(1)  # Пауза перед следующей попыткой
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 