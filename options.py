import argparse


def get_opt():
    
    parser = argparse.ArgumentParser()
    
    # Directory
    parser.add_argument('--input_img', type=str, default='./')
    parser.add_argument('--ground_truth', type=str, default='./')
    
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_change_ratio', type=float, default= 1.5)
    parser.add_argument('--learning_rate_policy_check_every', type=int, default=60)
    parser.add_argument('--learning_rate_slope_range', type=int, default=256)
    parser.add_argument('--min_learning_rate', type=float, default=9e-6)

    
    
    
    
    # Parameter
    parser.add_argument('--base_sf', type=float, default=1.0, help='base scale factor')
    parser.add_argument('--crop_size', type=int, default=128)
    
    parser.add_argument('--scale_factor', type=float, default=2.0)
    
    parser.add_argument('--min_iters', type=int, default=256)
    parser.add_argument('--run_test_every', type=int, default=50)
    opt = parser.parse_args()
    return opt

        
        