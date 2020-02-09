import face_recognition
import numpy as np
import cv2 as cv
import dlib


def face_detector_video_to_imgseq(video_path):
    video_capture = cv.VideoCapture()
    video_capture.open(video_path)
    fps = video_capture.get(cv.CAP_PROP_FPS)  # 视频帧率
    frames = video_capture.get(cv.CAP_PROP_FRAME_COUNT)  # 视频帧数量
    print('fps: {}, frames:{}'.format(fps, frames))
    cnn_detector = dlib.cnn_face_detection_model_v1('./res/mmod_human_face_detector.dat')
    folder_path = '.res/'

    # loop 15 to find the max roi region
    min_left, min_top, max_right, max_bottom = 1000, 1000, 0, 0
    for i in range(15):
        ret, frame = video_capture.read()
        # resize frame of the video to 1/4
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # convert image from BGR to RGB
        rgb_frame = small_frame[:, :, ::-1]
        # gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
        face_locations = cnn_detector(rgb_frame, 1)
        if face_locations[0].rect.left() < min_left:
            min_left = face_locations[0].rect.left()
        if face_locations[0].rect.top() < min_top:
            min_top = face_locations[0].rect.top()
        if face_locations[0].rect.right() > max_right:
            max_right = face_locations[0].rect.right()
        if face_locations[0].rect.bottom() > max_bottom:
            max_bottom = face_locations[0].rect.bottom()
    print('left:{}, top:{}, right:{}, bottom:{}'.format(min_left, min_top, max_right, max_bottom))

    for i in range(int(frames)):
        ret, frame = video_capture.read()
        if not frame is None:
            # view the detect results
            img = frame[min_top * 4: max_bottom * 4, min_left * 4: max_right * 4]
            img_dir = 'test' + str(i) + '.jpg'
            cv.imwrite(img_dir, img)

        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break

    video_capture.release()
    cv.destroyAllWindows()
    # return len(all_locations) / frames


if __name__ == '__main__':
    print(face_detector_video_to_imgseq('/Users/barackbao/PycharmProjects/face/RePSS/res/video1.mp4(3).avi'))
