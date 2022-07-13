from codebook import *
from encoder import *
from decoder import *


class VQGAN(nn.Module):
    def __int__(self, args):
        super(VQGAN, self).__int__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, images):
        encoding = self.encoder(images)
        conv_encoding = self.conv(encoding)
        codebook_mapping, codebook_indices, q_loss = self.codebook(conv_encoding)
        post_conv_mapping = self.post_conv(codebook_mapping)
        decoded_images = self.decoder(post_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, images):
        encoding = self.encoder(images)
        conv_encoding = self.conv(encoding)
        codebook_mapping, codebook_indices, q_loss = self.codebook(conv_encoding)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_conv_mapping = self.post_conv(z)
        decoded_images = self.decoder(post_conv_mapping)
        return decoded_images

    def calculate_lamda(self, perceptual_loss, gan_loss):
        last_layers = self.decoder.model(-1)
        last_layers_weight = last_layers.weigt
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layers_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layers_weight, retain_graph=True)[0]

        lamda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lamda = torch.clamp(lamda, 0, 1e4).detach()
        return 0.8 * lamda

    def adopt_weight(self,disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
