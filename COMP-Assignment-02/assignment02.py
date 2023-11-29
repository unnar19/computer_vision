import cv2
import time
import random
import numpy as np

def ransac(coords, delta, iterations):
    num_points = len(coords)
    if num_points >= 2:

        best_score = 0
        best_inside = []
        best_line = None

        for _ in range(iterations):

            looking = True
            while looking:
                x1, y1 = coords[random.randint(0,num_points-1)][0]
                x2, y2 = coords[random.randint(0,num_points-1)][0]
                if x1 != x2:
                    looking = False
            # y = mx + b
            m = (y2 - y1)/(x2 - x1)
            b = y1 - m * x1

            dist = np.array([np.abs(m * point[0][0] + b - point[0][1]) for point in coords])

            inside = coords[dist <= delta]

            score = inside.shape[0]/num_points

            if score > best_score:
                best_inside = inside
                best_score = score
                best_line = (m,b)

    return best_line, best_inside, best_score


cap = cv2.VideoCapture(0)

(w,h) = (int(cap.get(3)), int(cap.get(4)))

prev_time = time.time()

keep_score = []
keep_delta = []
keep_fps = []

print_score = 0
print_delta = 0
print_fps = 0

while(True):

    ret, frame = cap.read()

    edge = cv2.Canny(frame,70,300)

    white = cv2.findNonZero(edge)

    if type(white) != type(None):

        (m, b), best, score = ransac(white,0.9,40)

        if np.isnan(b) == False: 
            cv2.line(frame, (0,int(b)), (w,int(m*w+b)), (255,255,255), 2)
            for i in best:
                cv2.circle(frame, i[0], 2, (0,0,255), 1)
    else:
        score = 0


    time_delta = time.time() - prev_time
    prev_time = time.time()

    keep_score.append(score)
    keep_delta.append(time_delta)

    if len(keep_delta) == 30:
        print_score = np.mean(keep_score)
        print_delta = np.mean(keep_delta)*1000
        print_fps = 1/np.mean(keep_delta)

        keep_delta, keep_score = [], []

    cv2.putText(frame, "FPS:",(8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame, f"{print_fps:.2f}",(60,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame, "Delay:",(8,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame, f"{print_delta:.0f} ms",(60,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame, "Ratio:",(w-100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    cv2.putText(frame, f"{print_score:.3f}",(w-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1, cv2.LINE_AA)
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()