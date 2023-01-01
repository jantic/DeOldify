import logging
import base64
from io import BytesIO
from urllib.parse import urlparse

import cv2
import ffmpeg
import yt_dlp as youtube_dl
import requests

from fastai.core import *
from fastai.vision import *
from matplotlib.axes import Axes
from .filters import IFilter, MasterFilter, ColorizerFilter
from .generators import gen_inference_deep, gen_inference_wide
from PIL import Image
from IPython import display as ipython_display
from IPython.display import HTML
from IPython.display import Image as IpythonImage


# NOTE: adapted from https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/
def get_watermarked(pil_image: Image) -> Image:

    try:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        image = np.dstack([image, np.ones((h, w), dtype='uint8') * 255])
        pct = 0.05
        full_watermark = cv2.imread('./resource_images/watermark.png', cv2.IMREAD_UNCHANGED)
        (fwH, fwW) = full_watermark.shape[:2]
        w_height = int(pct * h)
        w_width = int((pct * h / fwH) * fwW)

        watermark = cv2.resize(full_watermark, (w_height, w_width), interpolation=cv2.INTER_AREA)
        overlay = np.zeros((h, w, 4), dtype='uint8')
        (w_height, w_width) = watermark.shape[:2]
        overlay[h - w_height - 10: h - 10, 10: 10 + w_width] = watermark

        # blend the two images together using transparent overlays
        output = image.copy()
        cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        final_image = Image.fromarray(rgb_image)
        return final_image

    except Exception:
        # Don't want this to crash everything, so let's just not watermark the image for now.
        return pil_image


class ModelImageVisualizer:
    def __init__(self,
                 filter_: IFilter,
                 results_dir: str = None):

        self.filter = filter_
        self.results_dir = None if results_dir is None else Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _clean_mem(self):
        torch.cuda.empty_cache()

    def _open_pil_image(self, path: Path) -> Image:
        return PIL.Image.open(path).convert('RGB')

    def _get_image_from_url(self, url: str) -> Image:
        response = requests.get(url,
                                timeout=30,
                                headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'}
                                )
        response_bytes = BytesIO(response.content)
        img = PIL.Image.open(response_bytes).convert('RGB')
        return img

    def plot_transformed_image_from_url(self,
                                        url: str,
                                        path: str = 'test_images/image.png',
                                        results_dir: Path = None,
                                        figure_size: Tuple[int, int] = (20, 20),
                                        render_factor: int = None,
                                        display_render_factor: bool = False,
                                        compare: bool = False,
                                        post_process: bool = True,
                                        watermarked: bool = True) -> Path:

        img = self._get_image_from_url(url)
        img.save(path)

        return self.plot_transformed_image(path=path,
                                           results_dir=results_dir,
                                           figure_size=figure_size,
                                           render_factor=render_factor,
                                           display_render_factor=display_render_factor,
                                           compare=compare,
                                           post_process=post_process,
                                           watermarked=watermarked)

    def plot_transformed_image(self,
                               path: str,
                               results_dir: Path = None,
                               figure_size: Tuple[int, int] = (20, 20),
                               render_factor: int = None,
                               display_render_factor: bool = False,
                               compare: bool = False,
                               post_process: bool = True,
                               watermarked: bool = True) -> Path:

        parsed_path = Path(path)

        if results_dir is None:
            results_dir = Path(self.results_dir)

        result_image = self.get_transformed_image(parsed_path,
                                                  render_factor,
                                                  post_process=post_process,
                                                  watermarked=watermarked)

        original_image = self._open_pil_image(parsed_path)

        if compare:
            self._plot_comparison(figure_size,
                                  render_factor,
                                  display_render_factor,
                                  original_image,
                                  result_image)
        else:
            self._plot_solo(figure_size,
                            render_factor,
                            display_render_factor,
                            result_image)

        original_image.close()
        result_path = self._save_result_image(parsed_path, result_image, results_dir=results_dir)
        result_image.close()

        return result_path

    def _plot_comparison(self,
                         figure_size: Tuple[int, int],
                         render_factor: int,
                         display_render_factor: bool,
                         orig: Image,
                         result: Image):
        fig, axes = plt.subplots(1, 2, figsize=figure_size)
        self._plot_image(
            orig,
            axes=axes[0],
            figure_size=figure_size,
            render_factor=render_factor,
            display_render_factor=False,
        )
        self._plot_image(
            result,
            axes=axes[1],
            figure_size=figure_size,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _plot_solo(self,
                   figure_size: Tuple[int, int],
                   render_factor: int,
                   display_render_factor: bool,
                   result: Image):

        fig, axes = plt.subplots(1, 1, figsize=figure_size)

        self._plot_image(
            result,
            axes=axes,
            figure_size=figure_size,
            render_factor=render_factor,
            display_render_factor=display_render_factor,
        )

    def _save_result_image(self,
                           source_path: Path,
                           result_image: Image,
                           results_dir=None) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)

        result_path = results_dir / source_path.name
        result_image.save(result_path)

        return result_path

    def get_transformed_image(self,
                              path: Path,
                              render_factor:
                              int = None,
                              post_process: bool = True,
                              watermarked: bool = True) -> Image:
        self._clean_mem()

        orig_image = self._open_pil_image(path)
        filtered_image = self.filter.filter(orig_image,
                                            orig_image,
                                            render_factor=render_factor,
                                            post_process=post_process)

        if watermarked:
            return get_watermarked(filtered_image)

        return filtered_image

    def _plot_image(self,
                    image: Image,
                    render_factor: int,
                    axes: Axes = None,
                    figure_size=(20, 20),
                    display_render_factor=False):

        if axes is None:
            _, axes = plt.subplots(figsize=figure_size)

        axes.imshow(np.asarray(image) / 255)
        axes.axis('off')

        if render_factor is not None and display_render_factor:
            plt.text(10,
                     10,
                     'render_factor: ' + str(render_factor),
                     color='white',
                     backgroundcolor='black')

    def _get_num_rows_columns(self, num_images: int, max_columns: int) -> Tuple[int, int]:
        columns = min(num_images, max_columns)
        rows = num_images // columns
        rows = rows if rows * columns == num_images else rows + 1

        return rows, columns


class VideoColorizer:

    def __init__(self, vis: ModelImageVisualizer):
        self.vis = vis

        work_folder = Path('./video')
        self.source_folder = work_folder / 'source'
        self.bw_frames_root = work_folder / 'bw_frames'
        self.audio_root = work_folder / 'audio'
        self.color_frames_root = work_folder / 'color_frames'
        self.result_folder = work_folder / 'result'

    def _purge_images(self, dir_path):
        logging.info('Purging *.jpg from %s directory...' % (str(dir_path)))

        for file_path in dir_path.iterdir():

            if not file_path.is_file():
                continue

            filename, file_extension = os.path.splitext(file_path)
            if file_extension.lower() == '.jpg':
                file_path.unlink()

        logging.info('Purge complete.')

    def _get_ffmpeg_probe(self, path: Path):
        try:
            probe = ffmpeg.probe(str(path))
            return probe
        except ffmpeg.Error as ex:
            logging.error('ffmpeg error: {0}'.format(ex), exc_info=True)
            logging.error('stdout:' + ex.stdout.decode('UTF-8'))
            logging.error('stderr:' + ex.stderr.decode('UTF-8'))
            raise ex
        except Exception as ex:
            logging.error('Failed to instantiate ffmpeg.probe.  Details: {0}'.format(ex), exc_info=True)
            raise ex

    def _get_video_stream_attributes(self, source_local_path: Path) -> dict:
        probe_result = self._get_ffmpeg_probe(source_local_path)
        video_streams = [stream
                         for stream
                         in probe_result['streams']
                         if stream['codec_type'] == 'video']

        video_streams_count = len(video_streams)
        if video_streams_count == 0:
            raise Exception('no video stream found')
        elif video_streams_count > 1:
            raise Exception('multiple video streams found')

        video_stream_attributes = video_streams[0]
        return video_stream_attributes

    def _download_video_from_url(self, source_url, target_path: Path):

        if target_path.exists():
            target_path.unlink()

        youtube_downloader_options = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': str(target_path),
            'retries': 30,
            'fragment-retries': 30
        }
        with youtube_dl.YoutubeDL(youtube_downloader_options) as youtube_downloader:
            youtube_downloader.download([source_url])

    def _extract_raw_frames(self,
                            source_path: Path,
                            expected_frame_count: int,
                            resume: bool = False):
        source_path_string = str(source_path)
        bw_frames_folder = self.bw_frames_root / source_path.stem
        bw_frame_path_template = str(bw_frames_folder / '%5d.jpg')
        bw_frames_folder.mkdir(parents=True, exist_ok=True)

        if resume:
            bw_folder_file_count = len([f for f in bw_frames_folder.iterdir() if os.path.isfile(f)])
            if bw_folder_file_count == expected_frame_count:
                logging.info('Resume set to TRUE: all raw frames found, skipping extraction of raw frames.')
                return

            logging.info('Resume set to TRUE: found %s frames but expected %s, proceeding to purge existing frames.' % (bw_folder_file_count, expected_frame_count))

        self._purge_images(bw_frames_folder)


        # NOTE: we specify "-vsync vfr" to prevent frame duplication, more in the link below:
        #       https://superuser.com/questions/1512575/why-total-frame-count-is-different-in-ffmpeg-than-ffprobe
        process = (
            ffmpeg
            .input(source_path_string)
            .output(bw_frame_path_template, vsync='vfr', format='image2', vcodec='mjpeg', **{'q:v': '0'})
            .global_args('-hide_banner')
            .global_args('-nostats')
            .global_args('-loglevel', 'error')
        )

        logging.info('Extracting raw frames (%s frames total)...' % expected_frame_count)

        try:
            process.run()
        except ffmpeg.Error as ex:
            logging.error('ffmpeg error: {0}'.format(ex), exc_info=True)
            logging.error('stdout:' + ex.stdout.decode('UTF-8'))
            logging.error('stderr:' + ex.stderr.decode('UTF-8'))
            raise ex
        except Exception as ex:
            logging.error('Error while extracting raw frames from source video. Details: {0}'.format(ex), exc_info=True)
            raise ex

        bw_folder_file_count = len([f for f in bw_frames_folder.iterdir() if os.path.isfile(f)])
        logging.info('Extraction complete, %s frames extracted.' % bw_folder_file_count)

    def _colorize_raw_frames(self,
                             source_path: Path,
                             render_factor:
                             int = None,
                             post_process: bool = True,
                             watermarked: bool = True,
                             resume: bool = False):

        bw_frames_folder = self.bw_frames_root / source_path.stem
        color_frames_folder = self.color_frames_root / source_path.stem
        color_frames_folder.mkdir(parents=True, exist_ok=True)

        if not resume:
            self._purge_images(color_frames_folder)

        logging.info('Colorizing frames...')

        all_source_files = [f for f in bw_frames_folder.iterdir() if os.path.isfile(f)]
        all_source_files = sorted(all_source_files, key=lambda a_file: int(os.path.splitext(a_file.name)[0]))

        if resume:
            remaining_source_files = [a_file
                                      for a_file
                                      in all_source_files
                                      if not os.path.isfile(color_frames_folder / a_file.name)]

            all_source_file_count = len(all_source_files)
            remaining_source_file_count = len(remaining_source_files)
            already_done_file_count = all_source_file_count - remaining_source_file_count

            logging.info('Resume set to TRUE: %s out of %s frames all ready done, %s frames remaining.' % (
                already_done_file_count,
                all_source_file_count,
                remaining_source_file_count))
        else:
            remaining_source_files = all_source_files

        for source_file in progress_bar(remaining_source_files):
            source_filename = source_file.name
            source_frame_path = bw_frames_folder / source_filename
            destination_frame_path = color_frames_folder / source_filename

            if os.path.isfile(source_frame_path):
                color_frame = self.vis.get_transformed_image(source_frame_path,
                                                             render_factor=render_factor,
                                                             post_process=post_process,
                                                             watermarked=watermarked)
                color_frame.save(str(destination_frame_path))

        logging.info('Colorization complete.')

    def _build_video(self, source_local_path: Path) -> Path:

        logging.info('Building video...')

        colorized_video_path = self.result_folder / (source_local_path.name.replace('.mp4', '_no_audio.mp4'))
        color_frames_folder = self.color_frames_root / source_local_path.stem
        color_frames_path_template = str(color_frames_folder / '%5d.jpg')

        colorized_video_path.parent.mkdir(parents=True, exist_ok=True)
        if colorized_video_path.exists():
            colorized_video_path.unlink()

        video_stream_attributes = self._get_video_stream_attributes(source_local_path)
        source_video_fps = video_stream_attributes['avg_frame_rate']

        process = (
            ffmpeg
            .input(color_frames_path_template, format='image2', vcodec='mjpeg', framerate=source_video_fps)
            .output(str(colorized_video_path), crf=17, vcodec='libx264')
            .global_args('-hide_banner')
            .global_args('-nostats')
            .global_args('-loglevel', 'error')
        )

        try:
            process.run()
        except ffmpeg.Error as e:
            logging.error('ffmpeg error: {0}'.format(e), exc_info=True)
            logging.error('stdout:' + e.stdout.decode('UTF-8'))
            logging.error('stderr:' + e.stderr.decode('UTF-8'))
            raise e
        except Exception as e:
            logging.error('Error while building output video.  Details: {0}'.format(e), exc_info=True)
            raise e

        result_path = self.result_folder / source_local_path.name

        if result_path.exists():
            result_path.unlink()

        # making copy of non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(colorized_video_path), str(result_path))

        # adding back sound here
        audio_file = Path(str(source_local_path).replace('.mp4', '.aac'))
        if audio_file.exists():
            audio_file.unlink()

        os.system('ffmpeg -y -i "%s" -vn -acodec copy "%s" -hide_banner -nostats -loglevel error' % (str(source_local_path), str(audio_file)))

        if audio_file.exists():
            os.system(
                'ffmpeg -y -i "%s" -i "%s" -shortest -c:v copy -c:a aac -b:a 256k "%s" -hide_banner -nostats -loglevel error' % (
                    str(colorized_video_path),
                    str(audio_file),
                    str(result_path)))

        logging.info('Build complete, video created here: %s' % str(result_path))

        return result_path

    def colorize_from_uri(self,
                          source_uri: str,
                          render_factor: int = None,
                          watermarked: bool = True,
                          post_process: bool = True,
                          resume: bool = False) -> Path:

        def is_url(x):
            try:
                result = urlparse(x)
                return all([result.scheme, result.netloc])
            except Exception:
                return False

        logging.info('Colorization starting')

        # if source_uri is a URL, download the target file and set filename accordingly.
        if is_url(source_uri):
            filename = os.path.basename(urlparse(source_uri).path)
            source_local_path = self.source_folder / filename
            self._download_video_from_url(source_uri, source_local_path)
        else:  # otherwise, source_uri is simply the filename of the source file without the local path.
            filename = source_uri
            source_local_path = self.source_folder / filename

        if not source_local_path.exists():
            raise Exception('video %s could not be found at specified location' % str(source_local_path))

        video_stream_attributes = self._get_video_stream_attributes(source_local_path)
        source_video_frame_count = int(video_stream_attributes['nb_frames'])

        self._extract_raw_frames(source_local_path,
                                 source_video_frame_count,
                                 resume=resume)

        self._colorize_raw_frames(source_local_path,
                                  render_factor=render_factor,
                                  post_process=post_process,
                                  watermarked=watermarked,
                                  resume=resume)

        result_path = self._build_video(source_local_path)

        logging.info('Colorization complete.')

        return result_path


def get_video_colorizer(render_factor: int = 21) -> VideoColorizer:
    logging.info('Initializing colorizer...')
    stable_video_colorizer = get_stable_video_colorizer(render_factor=render_factor)
    logging.info('Initialization complete.')

    return stable_video_colorizer


def get_artistic_video_colorizer(root_folder: Path = Path('./'),
                                 weights_name: str = 'ColorizeArtistic_gen',
                                 results_dir='result_images',
                                 render_factor: int = 35) -> VideoColorizer:

    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filter_ = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filter_, results_dir=results_dir)
    return VideoColorizer(vis)


def get_stable_video_colorizer(root_folder: Path = Path('./'),
                               weights_name: str = 'ColorizeVideo_gen',
                               results_dir='result_images',
                               render_factor: int = 21) -> VideoColorizer:

    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filter_ = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filter_, results_dir=results_dir)
    return VideoColorizer(vis)


def get_image_colorizer(root_folder: Path = Path('./'),
                        render_factor: int = 35,
                        artistic: bool = True) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)


def get_stable_image_colorizer(root_folder: Path = Path('./'),
                               weights_name: str = 'ColorizeStable_gen',
                               results_dir='result_images',
                               render_factor: int = 35) -> ModelImageVisualizer:

    learn = gen_inference_wide(root_folder=root_folder, weights_name=weights_name)
    filter_ = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filter_, results_dir=results_dir)

    return vis


def get_artistic_image_colorizer(root_folder: Path = Path('./'),
                                 weights_name: str = 'ColorizeArtistic_gen',
                                 results_dir='result_images',
                                 render_factor: int = 35) -> ModelImageVisualizer:

    learn = gen_inference_deep(root_folder=root_folder, weights_name=weights_name)
    filter_ = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    vis = ModelImageVisualizer(filter_, results_dir=results_dir)

    return vis


def show_image_in_notebook(image_path: Path):
    ipython_display.display(IpythonImage(str(image_path)))


def show_video_in_notebook(video_path: Path):
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    ipython_display.display(
        HTML(
            data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(
                encoded.decode('ascii')
            )
        )
    )
