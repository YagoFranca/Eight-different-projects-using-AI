import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Analisa o atual frame com a emoção
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)

        # Debugging: Retorna o resultado da estrutura
        print("Result:", result)

        # Verifica o reusltado da lista e seus elementos
        if isinstance(result, list) and len(result) > 0:
            # Extraia o primeiro resultado
            first_result = result[0]

            # Pega a atual emoção e verifica a atual confiança dele
            dominant_emotion = first_result['dominant_emotion']
            emotion_scores = first_result['emotion']

            # Filtra as emoções atuais - Feliz, Triste e Surpreso
            filtered_emotions = {key: emotion_scores[key] for key in ['happy', 'neutral', 'surprise'] if key in emotion_scores}

            # Mostra a atual emoção dominante
            cv2.putText(frame, f"Emocao: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Opicionalmente, Mostra no display as emoções capturadas alem da lista
            for i, (emotion, score) in enumerate(filtered_emotions.items()):
                text = f"{emotion}: {score:.2f}%"
                cv2.putText(frame, text, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            print("Sem face detectada.")

    except Exception as e:
        print(f"Error in DeepFace analysis: {e}")

    cv2.imshow('Emocao Detectada', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
