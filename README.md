## Representation Learning with Smooth Autoencoder

we propose a novel autoencoder variant, smooth autoencoder (SmAE), to learn robust and discriminative feature representations. Different from conventional autoencoders which reconstruct each sample from its encoding, we use the encoding of each sample to reconstruct its local neighbors. In this way, the learned representations are consistent among local neighbors and robust to small variations of the inputs. When trained with supervisory information, our approach forces samples from the same class to become more compact in the vicinity of data manifolds in the new representation space, where the samples are easier to be discriminated. Experimental results verify the effectiveness of the representations learned by our approach in image classification and face recognition tasks.

* This package depends on Mark Schmidt's freely available minFunc optimization package [link](http://www.cs.ubc.ca/~schmidtm/Software/minFunc/minFunc.html).

## Reference

If you use this code as part of any published research, please acknowledge the
following papers:

**"Representation Learning with Smooth Autoencoder."**  
Kongming Liang, Hong Chang, Zhen Cui, Shiguang Shan, Xilin Chen. *[ACCV](http://link.springer.com/chapter/10.1007/978-3-319-16808-1_6)*

	@incollection{liang2014representation,
	  title={Representation Learning with Smooth Autoencoder},
	  author={Liang, Kongming and Chang, Hong and Cui, Zhen and Shan, Shiguang and Chen, Xilin},
	  booktitle={Computer Vision--ACCV 2014},
	  pages={72--86},
	  year={2014},
	  publisher={Springer}
	}