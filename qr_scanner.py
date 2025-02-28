import cv2
import numpy as np
from pyzbar.pyzbar import decode
import requests
import time
import sys
import locale

# Устанавливаем кодировку для консоли
sys.stdout.reconfigure(encoding='utf-8')
# или
# locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

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

def detect_bright_spot(frame):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Увеличиваем размер ядра размытия для лучшей фильтрации шума
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Находим самую яркую точку
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    
    # Увеличиваем порог яркости
    brightness_threshold = maxVal - 30
    
    # Применяем адаптивную пороговую обработку
    thresh = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Применяем морфологические операции для удаления мелких бликов
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Фильтруем контуры по площади и круглости
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Вычисляем круглость контура
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Фильтруем по площади и круглости
            if area > 100 and area < 5000 and circularity > 0.7:
                filtered_contours.append(c)
        
        if filtered_contours:
            # Берем самый большой подходящий контур
            c = max(filtered_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Проверяем среднюю яркость внутри контура
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]
            
            if mean_brightness > 200:  # Проверка на достаточную яркость
                # Рисуем круг и координаты
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                text = f"x: {x:.2f}, y: {y:.2f}"
                cv2.putText(frame, text, (int(x) - 60, int(y) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Добавляем информацию о яркости
                brightness_text = f"brightness: {mean_brightness:.1f}"
                cv2.putText(frame, brightness_text, (int(x) - 60, int(y) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                return x, y, radius
    
    return None, None, None

def main():
    url = get_camera_url()
    print(f"Подключение к камере по адресу: {url}")
    
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Убедимся, что кадр имеет нужное разрешение
                    frame = cv2.resize(frame, (800, 600))
                    
                    # Обработка QR-кода
                    qr_x, qr_y, qr_data = process_qr(frame)
                    
                    # Обнаружение яркой точки
                    light_x, light_y, radius = detect_bright_spot(frame)
                    if light_x is not None:
                        print(f"Координаты источника света: ({light_x:.2f}, {light_y:.2f})")
                    
                    # Отображение кадра
                    cv2.imshow("QR Scanner", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Ошибка при получении кадра: {e}")
            time.sleep(1)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 