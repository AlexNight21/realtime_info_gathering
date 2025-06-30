from ultralytics import YOLO


class DetModel():
    def __init__(self, model="yolov8m.pt"):
        self.model = YOLO(model)
        # self.classes = self.model.names

    def get_det_info(self, frame):

        res_lst = []

        results = self.model(frame, verbose=False)
        for result in results:
            for dat in result.boxes:

                det_info = {
                    "bbox": list(map(int, dat.xyxy.tolist()[0])),
                    "conf_score": round(dat.conf.item(), 2),
                    "cls_id": int(dat.cls.item()),
                }

                res_lst.append(det_info)

        return res_lst
