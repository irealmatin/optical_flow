from argparse import ArgumentParser
import cv2
from sparse.lucas_kanade import lucas_kanade_method
from sparse.lucas_interactive import optimized_lucas_kanade

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["lucaskanade", "lucaskanade_interactive"],
        required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default="videos/car.mp4", help="Path to the video",
        
    ) 

    parser.add_argument(
        "--capture_index", type=int, default=0, help="Camera index for webcam input",
    )

    args = parser.parse_args()

    # Decide input source
    if args.video_path is not None:
        video_source = args.video_path
    else:
        video_source = args.capture_index

    if args.algorithm == "lucaskanade":
        lucas_kanade_method(video_source)
    elif args.algorithm == "lucaskanade_interactive":
        optimized_lucas_kanade(video_source)

if __name__ == "__main__":
    main()