# HW4: Segmentation

Your goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:
1. Given an image and sparse markings for foreground and background
2. Calculate SLIC over image
3. Calculate color histograms for all superpixels
4. Calculate color histograms for FG and BG
5. Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
6. Run a graph-cut algorithm to get the final segmentation

"Wow factor" bonus (20pt):
1. Make it interactive: Let the user draw the markings (carrying 0 pt for this part)
2. for every interaction step (mouse click, drag, etc.)
   1. recalculate only the FG-BG histograms,
   2. construct the graph and get a segmentation from the max-flow graph-cut,
   3. show the result immediately to the user (should be fast enough).

Due: Thu, Oct 26 9am

## Skeleton Code
HW4Segmentation.zip

You are given a lot of skeleton code to perform this task, so your part is relatively easy.
The following provided functions are essentially all you need to accomplish this:
```
centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)

fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)

norm_hists = normalize_histograms(color_hists)

graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
```

## Submission Details
Use the skeleton codes and provided inputs above.
Output is expected in a PNG image format: mask.png which contains the binary segmentation mask (0 = BG, 255 = FG).
See the example_output.png file to get an idea.

To get the bonus you need to show a working system. A 20-second screen recording will suffice.

Submission as always via Blackboard.
Make sure to have your name and ID proper on the submission zip filename.
Put all your codes for standard question into the skeleton codes main.py. Your program will be tested under VM using: `python main.py astronaut.png astronaut_marking.png ./`
A binary mask will be expected as output. Only change the main function to realize the required functionality.
(Get rid of all dependencies and disable all display functions please)

For bonus question, use a separate file `main_bonus.py`. Following command will be used to test your program:
```
python main_bonus.py astronaut.png
```
An interactive window should pop up and respond to mouse action with results.
You have the freedom to format `main_bonus.py` as appropriate but keep the interface intact.

Note that you need to have `main.py` for submission as well even if you do bonus question. Otherwise, you will get only 20 points for bonus provided that your results for bonus are good enough.

Like always, an example folder structure called "Samwise_Gamgee_111175657.zip" is uploaded under "Resources".
It is highly recommended to download that file, replace outputs with your own, zip it, and upload it.

## Grading criteria:

* Folder structure -5 points
* An example output with watermark is provided and its RMSD value to the key without watermark is 6.624. Details will be announced later
* Strict grading scheme will be applied for bonus question, only good enough results receive full credits.

You will not get extra credits implementing only the interactive interface (very easy with OpenCV API).

## Update
The video for bonus should be named as bonus.xxx. The format should be mainstream video format.

## Update
Here is how it should look like:

* First display a clean frame of astronaut.png
* Draw with mouse on that frame and the drawings must be visible (e.g. blue for foreground, red for background etc ..)
* When finish one mouse operation (e.g. finish a drag, a click), the result should pop up on another window showing the binary mask
* Remember you can only start the segmentation operation once you have at least one BG and one FG marking - to build the color histograms
* It is not required for the mask to update in real time with mouse position change, i.e. the unit for update is per mouse operation, not per mouse position change -- you may nevertheless experiment with real-time update, which would be cool ;)
* Show at least two different drawings of markings in the video
* Put the bonus video into Results folder along with mask.png

## About office hours:
I do not hold regular office hours. Office hours are by appointment through emails. Please refrain from sending emails asking to discuss the points you got for some homework or exam.
Unless it is a mistake, I will not create unfairness for the rest of the class by giving you more points without valid reasons.
