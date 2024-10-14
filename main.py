import cv2
import numpy as np
from art import tprint


def apply_yolo_object_detection(image_to_process):
    "Поиск на кадре людей"

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)

    #Модель начинает искать
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_scores, boxes = [], []
    objects_count = 0

    # Оценка найденных объектов
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            if class_index:
                continue
            if scores[class_index] > 0:
                center_x, center_y = int(obj[0] * width), int(obj[1] * height)
                obj_width, obj_height = int(obj[2] * width), int(obj[3] * height)
                boxes.append([center_x - obj_width // 2, center_y - obj_height // 2,
                              obj_width, obj_height])
                class_scores.append(float(scores[class_index]))

    #Отбор людей
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        image_to_process = draw_object_bounding_box(image_to_process, boxes[box_index])

    return image_to_process


def draw_object_bounding_box(image_to_process, box):
    "Отрисовка рамок для людей"

    font = cv2.FONT_HERSHEY_SIMPLEX
    final_image = cv2.rectangle(image_to_process, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
    final_image = cv2.putText(final_image, 'person', (box[0], box[1] - 10), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    return final_image


def start_video_object_detection(video: str):
    "Захват видео и запуск анализа"
    
    while True:
        try:
            #Читаем видео
            video = cv2.VideoCapture(video)
            out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while out.isOpened():
                ret, frame = video.read()

                if not ret:
                    tprint('Не удалось прочитать кадр')
                    return 0

                #Читаем кадр и изменяем согласно найденным объектам
                frame = apply_yolo_object_detection(frame)
                out.write(frame)

                #Отображаем кадр
                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video", frame)
                cv2.waitKey(1)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    return 0

            video.release()
            out.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':

    #Чтение весов YOLO
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    video = input("Path to video: ")
    start_video_object_detection(video)