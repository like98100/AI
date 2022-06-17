#환경 설정
#pip install opencv-python

import cv2
import mediapipe as mp
#print(cv2.__version__)

#def empty(pos):
#    pass

#이미지 출력
# img = cv2.imread('TestCat.jpg') # 해당 경로 파일 읽기
# imgColor = cv2.imread('TestCat.jpg', cv2.IMREAD_COLOR)          #컬러
# imgGray = cv2.imread('TestCat.jpg', cv2.IMREAD_GRAYSCALE)       #흑백
# imgUnchanged = cv2.imread('TestCat.jpg', cv2.IMREAD_UNCHANGED)  #투명 포함한 컬러

# name = "Trackbar"
# cv2.namedWindow(name)
# cv2.createTrackbar('threshold1', name, 0, 255, empty)   #min
# cv2.createTrackbar('threshold2', name, 0, 255, empty)   #max

# while True:
#     threshold1 = cv2.getTrackbarPos('threshold1', name)
#     threshold2 = cv2.getTrackbarPos('threshold2', name)
#     canny = cv2.Canny(img, threshold1, threshold2)  # 대상 이미지, minVal(하위 임계값), maxVal(상위 임계값)    픽셀의 임계값 > maxVal 일 시 경계선으로 간주
#
#     cv2.imshow(name, canny)  # 이미지 표시
#     cv2.imshow('imgWindow', img)  # 이미지 표시
#
#     if cv2.waitKey(1) == ord('e'):
#         break
#

#cv2.imshow('imgColor', imgColor)
#cv2.imshow('imgGray', imgGray)
#cv2.imshow('imgUnchanged', imgUnchanged)

##print(img.shape) # height, width, channel(RGB) 정보

#key = cv2.waitKey(0)                  # 지정된 시간(ms) 동안 사용자 키 입력 대기(입력 혹은 시간 경과 시 뒤 명령 실행)
#print(key)
#cv2.destroyAllWindows()         # 모든 창 제거

# result = cv2.imwrite('saveImg.jpg', canny);   # 이미지 저장
# print(result)

def overlay(image, x, y, w, h, overlay_image):   #대상 이미지(3채널), x, y, width, height, 덮어씌울 이미지(4채널)
    alpha = overlay_image[:,:,3] # BGRA
    mask_image = alpha / 255    # 0~1(불투명~완전)
    #(255,255) -> (1,1)
    #(255,0) -> (1,0)

    for c in range(0,3):    # channel BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:,:,c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))



mp_face_detection = mp.solutions.face_detection  # 얼굴 검줄을 위한 face_detection 모듈 사용
mp_drawing = mp.solutions.drawing_utils         # 얼굴의 특징을 그리기 위한 drawing_utils 모듈 사용

video = cv2.VideoCapture('video.mp4')

# 돼지코
image_left_ear = cv2.imread('Left Ear.png', cv2.IMREAD_UNCHANGED)
image_right_ear = cv2.imread('Right Ear.png', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('Nose.png', cv2.IMREAD_UNCHANGED)

# 소형 버전 돼지코
image_left_ear2 = cv2.imread('Left Ear2.png', cv2.IMREAD_UNCHANGED)
image_right_ear2 = cv2.imread('Right Ear2.png', cv2.IMREAD_UNCHANGED)
image_nose2 = cv2.imread('Nose2.png', cv2.IMREAD_UNCHANGED)

# 트롤 페이스
image_troll = cv2.imread('Troll face.png', cv2.IMREAD_UNCHANGED)

#   model_selection = 0 : 카메라 기준 근거리, 1 : 원거리  min_detection_confidence = 정확도가 float%면 얼굴로 확인
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.7) as face_detection:
    while video.isOpened():
        success, image = video.read()
        if not success:
            break

        image.flags.writeable = False                       # 수정불가
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # BGR을 RGB 형태로 변환
        results = face_detection.process(image)

        image.flags.writeable = True                        # 수정가능
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # RGB를 BGR로 변환

        if results.detections:                              # 검출된 얼굴이 있으면
            # 구분 특징 : 오른쪽/왼쪽 눈, 코 끝부분, 입 중심, 오른쪽/왼쪽 귀
            for detection in results.detections:            # 있는 얼굴만큼
#                mp_drawing.draw_detection(image, detection) # 도형 출력
                print(detection)

                #특정 위치 가져오기
                keypoints = detection.location_data.relative_keypoints
                right_ear = keypoints[0]    # 오른쪽 눈
                left_ear = keypoints[1]     # 왼쪽 눈
                nose_tip = keypoints[2]     # 코 끝부분

                h, w, _ = image.shape # height, width, channel

                # video 1

                # 돼지코
                right_ear = (int(right_ear.x * w) - 200, int(right_ear.y * h) - 175) # 이미지 내 실제 좌표
                left_ear = (int(left_ear.x * w) + 200, int(left_ear.y * h) - 175)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                # 소형 버전 돼지코
                # right_ear = (int(right_ear.x * w) - 75, int(right_ear.y * h) - 100) # 이미지 내 실제 좌표
                # left_ear = (int(left_ear.x * w) + 75, int(left_ear.y * h) - 100)
                # nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                # 트롤 페이스
                # nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h) - 75)


                # video 2 소형 돼지코
                # right_ear = (int(right_ear.x * w) - 75, int(right_ear.y * h) + 25)  # 이미지 내 실제 좌표
                # left_ear = (int(left_ear.x * w) - 15, int(left_ear.y * h) + 25)
                # nose_tip = (int(nose_tip.x * w) - 40, int(nose_tip.y * h) + 55)

                # Overlay(image, x,y,h,w,overlay_image)

                # 돼지코
                overlay(image, *right_ear, 225, 225, image_right_ear)
                overlay(image, *left_ear, 225, 225, image_left_ear)
                overlay(image, *nose_tip, 225, 225, image_nose)

                # 트롤 페이스
                # overlay(image, *nose_tip, 625, 500, image_troll)

                # 소형 버전 돼지코
                # overlay(image, *right_ear, 50, 50, image_right_ear2)
                # overlay(image, *left_ear, 50, 50, image_left_ear2)
                # overlay(image, *nose_tip, 150, 50, image_nose2)



        cv2.imshow('Face Detection', cv2.resize(image, None, fx = 0.3, fy = 0.3))

        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()