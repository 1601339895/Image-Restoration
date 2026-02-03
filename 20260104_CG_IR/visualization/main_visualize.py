import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(__file__))

from fig1_attention_overview import generate_fig1
from fig2_temperature_analysis import generate_fig2
from fig3_multideg_comparison import generate_fig3
from fig4_gating_visualization import generate_fig4

def main():
    parser = argparse.ArgumentParser(description='Generate all visualization figures for prompt-based adaptive attention')
    parser.add_argument('--adaptive_model', required=True, help='Path to adaptive model checkpoint')
    parser.add_argument('--baseline_model', required=True, help='Path to baseline model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory for figures')
    parser.add_argument('--num_images', type=int, default=50, help='Number of images per degradation type for statistics')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--figures', nargs='+', default=['1', '2', '3', '4'],
                       help='Which figures to generate (1, 2, 3, 4)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Prompt-Based Adaptive Attention Visualization")
    print("=" * 80)
    print(f"Adaptive Model: {args.adaptive_model}")
    print(f"Baseline Model: {args.baseline_model}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Figures to generate: {', '.join(args.figures)}")
    print("=" * 80)

    generated_figures = []
    total_time = 0

    # Figure 1: Attention Mechanism Overview
    if '1' in args.figures:
        print("\n[1/4] Generating Figure 1: Attention Mechanism Overview...")
        start_time = time.time()
        try:
            generate_fig1(args.adaptive_model, args.baseline_model, args.data_dir,
                         args.output_dir, args.device)
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"✓ Figure 1 completed in {elapsed:.2f}s")
            generated_figures.append("fig1_attention_overview.pdf")
        except Exception as e:
            print(f"✗ Figure 1 failed: {e}")

    # Figure 2: Temperature Coefficient Analysis
    if '2' in args.figures:
        print("\n[2/4] Generating Figure 2: Temperature Coefficient Analysis...")
        start_time = time.time()
        try:
            generate_fig2(args.adaptive_model, args.data_dir, args.output_dir,
                         args.num_images, args.device)
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"✓ Figure 2 completed in {elapsed:.2f}s")
            generated_figures.append("fig2_temperature_analysis.pdf")
        except Exception as e:
            print(f"✗ Figure 2 failed: {e}")

    # Figure 3: Multi-Degradation Attention Comparison
    if '3' in args.figures:
        print("\n[3/4] Generating Figure 3: Multi-Degradation Attention Comparison...")
        start_time = time.time()
        try:
            generate_fig3(args.adaptive_model, args.data_dir, args.output_dir, args.device)
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"✓ Figure 3 completed in {elapsed:.2f}s")
            generated_figures.append("fig3_multideg_comparison.pdf")
        except Exception as e:
            print(f"✗ Figure 3 failed: {e}")

    # Figure 4: Gating Mechanism Visualization
    if '4' in args.figures:
        print("\n[4/4] Generating Figure 4: Gating Mechanism Visualization...")
        start_time = time.time()
        try:
            generate_fig4(args.adaptive_model, args.data_dir, args.output_dir, args.device)
            elapsed = time.time() - start_time
            total_time += elapsed
            print(f"✓ Figure 4 completed in {elapsed:.2f}s")
            generated_figures.append("fig4_gating_visualization.pdf")
        except Exception as e:
            print(f"✗ Figure 4 failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Generated {len(generated_figures)} figures:")
    for fig in generated_figures:
        fig_path = os.path.join(args.output_dir, fig)
        print(f"  - {fig_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()
