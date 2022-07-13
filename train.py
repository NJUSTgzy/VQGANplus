
from VQGAN import VQGAN
from utils import *
from discriminator import Discriminator
import torch
from tqdm import tqdm
import argparse


class Train:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.divice)
        self.discriminator = Discriminator(args).to(device=args.divice)
        self.discriminator.apply(weight_init())
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, eps=1e-8,
            betas=(args.beta1, args.beta2)
        )

        return opt_vq, self.opt_disc

    def prepare_training(self):
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoder_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoder_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i,
                                                          threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoder_images)
                    rec_loss = torch.abs(imgs - decoder_images)
                    perceptual_res_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_res_loss = perceptual_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    lamda = self.vqgan.calculate_lamda(perceptual_res_loss, g_loss)
                    vq_loss = perceptual_res_loss + q_loss + disc_factor * lamda * g_loss

                    self.opt_vq.zero_grad()
                    vq_loss.backward()
                    self.opt_vq.step()

                    if i % 100 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoder_images.add(1).mul(0.5)[:4]))
                            torch.save_images(real_fake_images, os.path.join("result", f"{epoch}_{i}.jpg"), nrow=4)

                    pbar.set_postfix(

                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num--codebook--vectors', type=int, default=256 * 4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=3)
    parser.add_argument('--dataset-path', type=str, default='/data')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--disc-start', type=int, default=1000)
    parser.add_argument('--disc-factor', type=float, default=1.)
    parser.add_argument('--l2-loss-factor', type=float, default=1.)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.)

    args = parser.parse_args()
    args.dataset_path = ""
    train_vqgan = Train(args)
