import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
import glob
import os

"""
Download youtube video, extract audio, and split into 1-second chunks
"""
def get_vid(link):
    # Download video and extract audio using youtube-dl
    subprocess.call('youtube-dl -x --audio-format "wav" -o ~/music-semantics/data/"%(title)s.%(ext)s" ' + link,
                    shell=True)

    file_list = glob.glob('data/*.wav')
    latest_file = max(file_list, key=os.path.getctime)

    audio = AudioSegment.from_file(latest_file)
    # Split into 1-second chunks
    chunks = make_chunks(audio, 1000)

    video_title = latest_file.split('/')[1][:-4]
    video_title = video_title.replace(' ', '-')
    if not os.path.exists('data/' + video_title):
        os.makedirs('data/' + video_title)

    for i, chunk in enumerate(chunks):
        chunk_name = video_title + '-{0}.wav'.format(i)
        chunk.export('data/' + video_title + '/' + chunk_name, format='wav')

    # Clean up, remove full wav file
    os.remove(latest_file)

"""
Generate NSynth audio embeddings
Assumes WaveNet checkpoint is in wavenet-ckpt/model.ckpt-200000
"""
def make_embeddings(folder):
    if not os.path.exists(folder + '-embed')
        os.makedirs(folder + '-embed')
    
    subprocess.call(
        'nsynth_save_embeddings --checkpoint_path=wavenet-ckpt/model.ckpt-200000 --source_path=' + folder +
        ' --save_path=' + folder + '-embed --batch_size=4', shell=True
    )
