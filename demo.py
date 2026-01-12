from argparse import ArgumentParser
from sparse.lucas_kanade import lucaskanade_method , lucaskanade_manual_tracking
from sparse.lucaskanade_interactive import optimized_lucaskanade_method
from Dence.dence_pyramid import dence_method
from Dence.RLOF import rlof_method

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["lucaskanade", "lucaskanade_interactive" , "lucaskanade_manual , dence_pyramid_lk" , "dence_rlof"],
        required=True,
        help="Optical flow algorithm to use",
    )
    parser.add_argument(
        "--video_path", default=None, help="Path to the video",
        
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

    # Decide for algorithm 
    if args.algorithm == "lucaskanade":
        lucaskanade_method(video_source)

    elif args.algorithm == "lucaskanade_manual":
        lucaskanade_manual_tracking(video_source)

    elif args.algorithm == "lucaskanade_interactive":
        optimized_lucaskanade_method(video_source)

    elif args.algorithm == " dence_pyramid_lk":
        dence_method(video_source)

    elif args.algorithm == "dence_rlof":
        rlof_method(video_source)

if __name__ == "__main__":
    main()