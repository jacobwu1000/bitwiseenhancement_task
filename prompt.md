Task: Overlay a background image over the white areas of a logo using bitwise operations.

You are given:
- A background image (`background.png`)
- Several logo images (`logo_1.png`, `logo_2.png`, `logo_3.png`) located in the `input/` directory of this task

Each logo contains white regions that should be replaced with the background. 

Your task:
For each logo, generate an output image in your workspace with the same size as the original logo and with the white regions of the logo replaced by the background image. The background image should show through wherever the logo is white, while the rest of the logo remains unchanged.

Output files:
- `logo_output_1.png`
- `logo_output_2.png`
- `logo_output_3.png`

Notes:
- You must use OpenCV2's bitwise operations 
- Both the logos and the background are square images. You may need to resize background image to match the size of each logo.
