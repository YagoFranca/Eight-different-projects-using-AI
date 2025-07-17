from ultralytics import YOLO

model = YOLO(r'C:\Users\yago_\VisualCodeProjects\tectime\IA\Yolo\yolo11n.pt')

model.predict(source='0', show = True, classes=[0])

"Contagem de estoque para produtos (POS) na fabrica"
