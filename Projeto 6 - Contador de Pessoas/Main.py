from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Data/Videos/Contador de Pessoas/people.mp4")  # For Video


# Função para seleção interativa das linhas
def select_lines(cap):
    # Lê o primeiro frame do vídeo
    success, img = cap.read()
    if not success:
        print("Erro ao ler o vídeo")
        exit()

    points = []

    # Função de callback do mouse
    def mouse_callback(event, x, y, flags, param):
        nonlocal img, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if len(points) % 2 == 0:  # Desenha linha quando temos dois pontos
                cv2.line(img, points[-2], points[-1], (0, 0, 255), 2)
            cv2.imshow("Selecione as linhas", img)

    cv2.namedWindow("Selecione as linhas")
    cv2.setMouseCallback("Selecione as linhas", mouse_callback)

    # Instruções interativas
    print("Selecione duas linhas:")
    print("1. Clique em dois pontos para a linha superior")
    print("2. Clique em dois pontos para a linha inferior")
    print("Pressione Q para confirmar após selecionar 4 pontos")

    while True:
        display_img = img.copy()
        # Adiciona instruções textuais
        if len(points) < 2:
            text = "Linha Superior: selecione dois pontos"
        elif len(points) < 4:
            text = "Linha Inferior: selecione dois pontos"
        else:
            text = "Pressione Q para continuar"

        cv2.putText(display_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Selecione as linhas", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= 4:
            break

    cv2.destroyAllWindows()
    return points


# Seleciona as linhas interativamente
selected_points = select_lines(cap)

# Verifica se temos pontos suficientes
if len(selected_points) != 4:
    print("Selecione exatamente 4 pontos (2 para cada linha)")
    exit()

# Define os limites a partir dos pontos selecionados
limitsUp = [selected_points[0][0], selected_points[0][1],
            selected_points[1][0], selected_points[1][1]]
limitsDown = [selected_points[2][0], selected_points[2][1],
              selected_points[3][0], selected_points[3][1]]

# Reinicia o vídeo do início
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Restante do código original...
model = YOLO("../IA/Yolo/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('../Data/Images/Contador de Pessoas/mask.png')

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread(r"C:\Users\yago_\VisualCodeProjects\Projetos\Data\Images\Contador de Pessoas\graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    # # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()