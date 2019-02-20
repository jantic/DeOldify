# DeOldify

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/jantic/DeOldify/blob/master/ImageColorizerColab.ipynb) 

[Get more updates on Twitter <img src="result_images/Twitter_Social_Icon_Rounded_Square_Color.svg" width="16">](https://twitter.com/citnaj)


Simply put, the mission of this project is to colorize and restore old images.  I'll get into the details in a bit, but first let's get to the pictures!  BTW – most of these source images originally came from the [TheWayWeWere](https://www.reddit.com/r/TheWayWeWere) subreddit, so credit to them for finding such great photos.


#### Some of many results - These are pretty typical!

Maria Anderson as the Fairy Fleur de farine and Lyubov Rabtsova as her page in the ballet “Sleeping Beauty” at the Imperial Theater, St. Petersburg, Russia, 1890.

![Ballerinas](result_images/Ballerinas.jpg)

Woman relaxing in her livingroom (1920, Sweden)

![SwedenLivingRoom](result_images/SweedishLivingRoom1920.jpg)

Medical Students pose with a cadaver around 1890

![MedStudents](result_images/MedStudentsCards.jpg)

Surfer in Hawaii, 1890

![1890Surfer](result_images/1890Surfer.jpg)

Whirling Horse, 1898

![WhirlingHorse](result_images/WhirlingHorse.jpg)

Interior of Miller and Shoemaker Soda Fountain, 1899

![SodaFountain](result_images/SodaShop.jpg)

Paris in the 1880s

![Paris1880s](result_images/Paris1880s.jpg)

Edinburgh from the sky in the 1920s

![Edinburgh](result_images/FlyingOverEdinburgh.jpg)

Texas Woman in 1938

![TexasWoman](result_images/TexasWoman.jpg)

People watching a television set for the first time at Waterloo station, London, 1936

![Television](result_images/FirstTV1930s.jpg)

Geography Lessons in 1850

![Geography](result_images/GeographyLessons.jpg)

Chinese Opium Smokers in 1880

![OpiumReal](result_images/ChineseOpium1880s.jpg)


#### Note that even really old and/or poor quality photos will still turn out looking pretty cool:

Deadwood, South Dakota, 1877

![Deadwood](result_images/OldWest.jpg)

Siblings in 1877

![Deadwood](result_images/Olds1875.jpg)

Portsmouth Square in San Franscisco, 1851

![PortsmouthSquare](result_images/SanFran1850sRetry.jpg)

Samurais, circa 1860s

![Samurais](result_images/Samurais.jpg)

#### Granted, the model isn't always perfect.  This one's red hand drives me nuts because it's otherwise fantastic:

Seneca Native in 1908

![Samurais](result_images/SenecaNative1908.jpg)

#### It can also colorize b&w line drawings:

![OpiumDrawing](result_images/OpiumSmokersDrawing.jpg)


### The Technical Details

This is a deep learning based model.  More specifically, what I've done is combined the following approaches:
* **Self-Attention Generative Adversarial Network** (https://arxiv.org/abs/1805.08318).  Except the generator is a **pretrained U-Net**, and I've just modified it to have the spectral normalization and self-attention.  It's a pretty straightforward translation. I'll tell you what though – it made all the difference when I switched to this after trying desperately to get a Wasserstein GAN version to work.  I liked the theory of Wasserstein GANs but it just didn't pan out in practice.  But I'm in *love* with Self-Attention GANs.
* Training structure inspired by (but not the same as) **Progressive Growing of GANs** (https://arxiv.org/abs/1710.10196).  The difference here is the number of layers remains constant – I just changed the size of the input progressively and adjusted learning rates to make sure that the transitions between sizes happened successfully.  It seems to have the same basic end result – training is faster, more stable, and generalizes better.  
* **Two Time-Scale Update Rule** (https://arxiv.org/abs/1706.08500).  This is also very straightforward – it's just one to one generator/critic iterations and higher critic learning rate. 
* **Generator Loss** is two parts:  One is a basic Perceptual Loss (or Feature Loss) based on VGG16 – this just biases the generator model to replicate the input image.  The second is the loss score from the critic.  For the curious – Perceptual Loss isn't sufficient by itself to produce good results.  It tends to just encourage a bunch of brown/green/blue – you know, cheating to the test, basically, which neural networks are really good at doing!  Key thing to realize here is that GANs essentially are learning the loss function for you – which is really one big step closer to toward the ideal that we're shooting for in machine learning.  And of course you generally get much better results when you get the machine to learn something you were previously hand coding.  That's certainly the case here.

The beauty of this model is that it should be generally useful for all sorts of image modification, and it should do it quite well.  What you're seeing above are the results of the colorization model, but that's just one component in a pipeline that I'm looking to develop here with the exact same model. 

What I develop next with this model will be based on trying to solve the problem of making these old images look great, so the next item on the agenda for me is the "defade" model.  I've committed initial efforts on that and it's in the early stages of training as I write this.  Basically it's just training the same model to reconstruct images that augmented with ridiculous contrast/brightness adjustments, as a simulation of fading photos and photos taken with old/bad equipment. I've already seen some promising results on that as well:

![DeloresTwoChanges](result_images/DeloresTwoChanges.jpg)

### This Project, Going Forward
So that's the gist of this project – I'm looking to make old photos look reeeeaaally good with GANs, and more importantly, make the project *useful*.  And yes, I'm definitely interested in doing video, but first I need to sort out how to get this model under control with memory (it's a beast).  It'd be nice if the models didn't take two to three days to train on a 1080TI as well (typical of GANs, unfortunately). In the meantime though this is going to be my baby and I'll be actively updating and improving the code over the foreseeable future.  I'll try to make this as user-friendly as possible, but I'm sure there's going to be hiccups along the way.  

Oh and I swear I'll document the code properly...eventually.  Admittedly I'm *one of those* people who believes in "self documenting code" (LOL).

### Getting Started Yourself
The easiest way to get started is to simply try out colorization here on Colab: https://colab.research.google.com/github/jantic/DeOldify/blob/master/DeOldify_colab.ipynb.  This was contributed by Matt Robinson, and it's simply awesome.


#### Hardware and Operating System Requirements

* **(Training Only) BEEFY Graphics card**.  I'd really like to have more memory than the 11 GB in my GeForce 1080TI (11GB).  You'll have a tough time with less.  The Unet and Critic are ridiculously large but honestly I just kept getting better results the bigger I made them.  
* **(Colorization Alone) A decent graphics card**. You'll benefit from having more memory in a graphics card in terms of the quality of the output achievable by.  Now what the term "decent" means exactly...I'm going to say 6GB +.  I haven't tried it but in my head the math works....  
* **Linux (or maybe Windows 10)**  I'm using Ubuntu 16.04, but nothing about this precludes Windows 10 support as far as I know.  I just haven't tested it and am not going to make it a priority for now.  

#### Easy Install

You should now be able to do a simple install with Anaconda. Here are the steps:

Open the command line and navigate to the root folder you wish to install.  Then type the following commands 
```console
git clone https://github.com/jantic/DeOldify.git DeOldify
cd DeOldify
conda env create -f environment.yml
```
Then start running with these commands:
```console
source activate deoldify
jupyter lab
```

From there you can start running the notebooks in Jupyter Lab, via the url they provide you in the console.  

**Disclaimer**: This conda install process is new- I did test it locally but the classic developer's excuse is "well it works on my machine!" I'm keeping that in mind- there's a good chance it doesn't necessarily work on others's machines!  I probably, most definitely did something wrong here.  Definitely, in fact.  Please let me know via opening an issue. Pobody's nerfect.

#### More Details for Those So Inclined

This project is built around the wonderful Fast.AI library.  Unfortunately, it's the -old- version and I have yet to upgrade it to the new version.  (That's definitely [update 11/18/2018: maybe] on the agenda.)  So prereqs, in summary:
* ***Old* Fast.AI library (version 0.7)** [**UPDATE 11/18/2018**] A forked version is now bundled with the project, for ease of deployment and independence from whatever happens to the old version from here on out.
* **Python 3.6**
* **Pytorch 0.4.1** (needs spectral_norm, so  latest stable release is needed). https://pytorch.org/get-started/locally/
* **Jupyter Lab** `conda install -c conda-forge jupyterlab`
* **Tensorboard** (i.e. install Tensorflow) and **TensorboardX** (https://github.com/lanpa/tensorboardX).  I guess you don't *have* to but man, life is so much better with it.  And I've conveniently provided hooks/callbacks to automatically write all kinds of stuff to tensorboard for you already!  The notebooks have examples of these being instantiated (or commented out since I didn't really need the ones doing histograms of the model weights).  Notably, progress images will be written to Tensorboard every 200 iterations by default, so you get a constant and convenient look at what the model is doing.  `conda install -c anaconda tensorflow-gpu` 
* **ImageNet** – Only if training of course. It proved to be a great dataset.  http://www.image-net.org/download-images

### Pretrained Weights 
To start right away with your own images without training the model yourself, [download the weights here](https://www.dropbox.com/s/3e4dqky91h824ik/colorize_gen.pth) (right click and download from this link). Then open the [ColorizeVisualization.ipynb](ColorizeVisualization.ipynb) in Jupyter Lab.  Make sure that there's this sort of line in the notebook referencing the weights:
```python
colorizer_path = IMAGENET.parent/('colorize_gen_192.h5')
```

Then you simply pass it to this (all this should be in the notebooks already):
```python
filters = [Colorizer(gpu=0, weights_path=colorizer_path)]
```

Which then feed into this:

```python
vis = ModelImageVisualizer(filters, render_factor=render_factor, results_dir='result_images')
```

### Colorizing Your Own Photos
Just drop whatever images in the `/test_images/` folder you want to run this against and you can visualize the results inside the notebook with lines like this:

```python
vis.plot_transformed_image("test_images/derp.jpg")
```

The result images will automatically go into that **result_dir** defined above, in addition to being displayed in Jupyter.

There's a **render_factor** variable that basically determines the quality of the rendered colors (but not the resolution of the output image).  The higher it is, the better, but you'll also need more GPU memory to accomodate this.  The max I've been able to have my GeForce 1080TI use is 42.  Lower the number if you get a CUDA_OUT_OF_MEMORY error.  You can customize this render_factor per image like this, overriding the default:

```python
vis.plot_transformed_image("test_images/Chief.jpg", render_factor=17)
```

For older and low quality images in particular, this seems to improve the colorization pretty reliably.  In contrast, more detailed and higher quality images tend to do better with a higher render_factor.

### Additional Things to Know

Model weight saves are also done automatically during the training runs by the `GANTrainer` – defaulting to saving every 1000 iterations (it's an expensive operation).  They're stored in the root training data folder you provide, and the name goes by the save_base_name you provide to the training schedule.  Weights are saved for each training size separately.

I'd recommend navigating the code top down – the Jupyter notebooks are the place to start.  I treat them just as a convenient interface to prototype and visualize – everything else goes into `.py` files (and therefore a proper IDE) as soon as I can find a place for them.  I already have visualization examples conveniently included – just open the `xVisualization` notebooks to run these – they point to test images already included in the project so you can start right away (in test_images). 

The "GAN Schedules" you'll see in the notebooks are probably the ugliest looking thing I've put in the code, but they're just my version of implementing progressive GAN training, suited to a Unet generator.  That's all that's going on there really.

[Pretrained weights for the colorizer generator again are here](https://www.dropbox.com/s/3e4dqky91h824ik/colorize_gen.pth) (right click and download from this link). The DeFade stuff is still a work in progress so I'll try to get good weights for those up in a few days.

Generally with training, you'll start seeing good results when you get midway through size 192px (assuming you're following the progressive training examples I laid out in the notebooks).  Note that this training regime is still a work in progress- I'm stil trying to figure out what exactly is optimal.  In other words, there's a good chance you'll find something to improve upon there.

I'm sure I screwed up something putting this up, so [please let me know](https://github.com/jantic/DeOldify/issues/new) if that's the case. 

### Known Issues

* Getting the best images really boils down to the **art of selection**.  You'll mostly get good results the first go, but playing around with the render_factor a bit may make a difference.  Thus, I'd consider this tool at this point fit for the "AI artist" but not something I'd deploy as a general purpose tool for all consumers.  It's just not there yet. 
* The model *loves* blue clothing.  Not quite sure what the answer is yet, but I'll be on the lookout for a solution!

### Want More?

I'll be posting more results on Twitter. [<img src="result_images/Twitter_Social_Icon_Rounded_Square_Color.svg" width="28">](https://twitter.com/citnaj)

---

### UPDATE 11/15/2018
I just put up a bunch of significant improvements!  I'll just repeat what I put in Twitter, here:

So first, this image should really help visualize what is going on under the hood. Notice the smallified square image in the center.

![BeforeAfterChief](result_images/BeforeAfterChief.jpg)


#### Squarification 
That small square center image is what the deep learning generator actually generates now.  Before I was just shrinking the images keeping the same aspect ratio.  It turns out, the model does better with squares- even if they're distorted in the process!

Note that I tried other things like keeping the core image's aspect ratio the same and doing various types of padding to make a square (reflect, symmetric, 0, etc).  None of this worked as well.  Two reasons why I think this works.  

* One- model was trained on squares;
* Two- at smaller resolutions I think this is particularly significant- you're giving the model more real image to work with if you just stretch it as opposed to padding.  And padding wasn't something the model trained on anyway.

#### Chrominance Optimization
It turns out that the human eye doesn't perceive color (chrominance) with nearly as much sensitivity as it does intensity (luminance).  Hence, we can render the color part at much lower resolution compared to the desired target res.

Before, I was having the model render the image at the same size as the end result image that you saw. So you maxed out around 550px (maybe) because the GPU couldn't handle anymore.  Now?  Colors can be rendered at say a tiny 272x272 (as the image above), then the color part of the model output is simply resized and stretched to map over the much higher resolution original images's luminance portion (we already have that!). So the end result looks fantastic, because your eyes can't tell the difference with the color anyway!

#### Graceful Rendering Degradation
With the above, we're now able to generate much more consistently good looking images, even at different color gpu rendering sizes.  Basically, you do generally get a better image if you have the model take up more memory with a bigger render.  BUT if you reduce that memory footprint even in half with having the model render a smaller image, the difference in image quality of the end result is often pretty negligible.  This effectively means the colorization is usable on a wide variety of machines now! 

i.e. You don't need a GeForce 1080TI to do it anymore.  You can get by with much less.

#### Consistent Rendering Quality 
Finally- With the above, I was finally able to narrow down a scheme to make it so that the hunt to find the best version of what the model can render is a lot less tedious.  Basically, it amounts to providing a render_factor (int) by the user and multiplying it by a base size multiplier of 16.  This, combined with the square rendering, plays well together.  It means that you get predictable behavior of rendering as you increase and decrease render_factor, without too many surprise glitches.

Increase render_factor: Get more details right.  Decrease:  Still looks good but might miss some details.  Simple!  So you're no longer going to deal with a clumsy sz factor.  Bonus:  The memory usage is consistent and predictable so you just have to figure out the render_factor that works for your gpu once and forget about it.  I'll probably try to make that render_factor determination automatic eventually but this should be a big improvement in the meantime.

#### P.S 

You're not losing any image anymore with padding issues.  That's solved as a byproduct.  

#### Also Also
I added a new generic filter interface that replaces the visualizer dealing with models directly.  The visualizer loops through these filters that you provide as a list.  They don't have to be backed by deep learning models- they can be any image modification you want!
