import cv2
import time
import os

#os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp;buffer_size;2'
#cap = cv2.VideoCapture('rtsp://192.168.1.38/video', cv2.CAP_FFMPEG)

cap = cv2.VideoCapture(0)

(w,h) = (int(cap.get(3)), int(cap.get(4)))

prev_frame = 0
frame_count = 0
prev_time = time.time()
fps = 30
mean_fps = []

while(True):
    frame_count += 1

    ret, frame = cap.read()

    whitest = 0
    whitest_loc = (0,0)

    reddest = 0
    reddest_loc = (0,0)

    frame3 = frame/3

    for y in range(h):
        for x in range(w):
            r = frame3[y,x][2]
            g = frame3[y,x][1]
            b = frame3[y,x][0]

            white = r + g + b
            if white > whitest:
                whitest_loc = (x, y)
                whitest = white

            red = (r - b) + (r - g)
            if red >= reddest:
                reddest_loc = (x, y)
                reddest = red

    time_delta = time.time() - prev_time
    if time_delta >= 1:
        prev_time = time.time()
        fps = frame_count/time_delta
        mean_fps.append(fps)
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.2f}",(8,32), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}",(8,32), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2, cv2.LINE_AA)
    cv2.circle(frame, whitest_loc, 20, (0,0,0), 3)
    cv2.circle(frame, reddest_loc, 20, (0,0,255), 3)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

meanfps = sum(mean_fps)/len(mean_fps)
print(f"  mean fps: {meanfps:.2f}")
print(f"mean delay: {1/meanfps:.2f}")

cap.release()
cv2.destroyAllWindows()