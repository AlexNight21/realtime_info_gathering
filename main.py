from src.camera.camera import Camera
from src.det_model.det_model import DetModel
import torch




def main():

    det_model = DetModel()
    camera = Camera(det_model=det_model)

    camera.get_frames()

    del det_model
    
    #clear cuda cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
