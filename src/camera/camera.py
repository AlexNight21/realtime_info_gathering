import cv2


class Camera():
    def __init__(self, det_model, camera_id=0):
        self.camera_id = camera_id
        self.det_model = det_model

    # def _set_camera_params(self, cap):
    #     pass

    def show_det_info(self, frame, det_info):

        for dat in det_info:

            tl = (dat["bbox"][0], dat["bbox"][1])
            br = (dat["bbox"][2], dat["bbox"][3])
            
            frame  = cv2.rectangle(frame , tl, br, (0, 255, 0), 1)
            frame = cv2.putText(frame, f"cls: {dat['cls_id']}, scr: {dat['conf_score']}", tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        return frame

    def get_frames(self):

        cap = cv2.VideoCapture(self.camera_id)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            det_info = self.det_model.get_det_info(frame)
            frame = self.show_det_info(frame, det_info)

            cv2.imshow("video feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Exiting video feed")
                break

        cap.release()
        cv2.destroyAllWindows()
