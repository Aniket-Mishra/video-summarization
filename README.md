# video-summarization
Summarizing large original videos into small summarized videos
## Generate summary of any video

Generate a summary of any video through its subtitles.

This is the community driven approach towards the summarization by the **[OpenGenus](https://github.com/opengenus)** community.

# Installing vidsum

In order to install vidsum, simply clone the repository to a local directory. You can do this by running the following commands:
```sh
$ git clone https://github.com/OpenGenus/vidsum.git

$ cd vidsum/code

```
Please note that vidsum requires the following packages to be installed:

- certifi
- chardet
- decorator
- docopt
- idna
- imageio
- lxml
- moviepy
- nltk
- numpy
- olefile
- Pillow
- pysrt
- pytube
- requests
- six
- sumysion)
- tqdm
- urllib3
- youtube-dl

If you do not have these packages installed, then you can do so by running this command:
```sh
$ pip install -r requirements.txt

```

# Usage

To generate summary of a video file `sample.mp4` with subtitle file `subtitle.srt` :
```python
python sum.py -i sample.mp4 -s subtitle.srt
```
To summarize a YouTube video from its url:
```python
python sum.py -u <url>
```
If you want to remain the downloaded YouTube video and subtitles:
```python
python sum.py -u <url> -k
```
