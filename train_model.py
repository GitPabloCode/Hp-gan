
import numpy as np
import random as rnd
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from braniac.readers.body import SequenceBodyReader
from braniac.format import SourceFactory
from utils import gradient_penalty
from visualize_skeleton import Visualize_skeleton



'''
Main entry point that drive GAN training for body and skeleton data.

Args:
args: arg parser object, contains all arguments provided by the user.
'''

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        # Forward propagate RNN
        out, _ = self.rnn(x)  
    
        # out: tensor of shape (batch_size, hidden_size)
        # out: (n, 512)
        
        out = self.fc(out)
        # out: (n, 1500)
        out = out.reshape(-1, 20, 75)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), inplace=False)
        x = nn.functional.leaky_relu(self.fc2(x), inplace=False)
        out = nn.functional.sigmoid(self.fc3(x))
        return out

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), inplace=False)
        x = nn.functional.leaky_relu(self.fc2(x), inplace=False)
        out = self.fc3(x)
        return out


max_epochs = 300
dataset = "nturgbd"
source = SourceFactory(dataset, 'cameras.h5')
sensor = source.create_sensor()
body_inf = source.create_body()
visualizer = Visualize_skeleton(sensor, body_inf)

# input and output information
input_sequence_length = 10
output_sequence_length = 20
sequence_length = input_sequence_length + output_sequence_length
input_size = 75
input_discriminator_size = 2250
output_size = 1500
num_layers = 2
inputs_depth = 512
z_size = 128 # Latent value that control predicted poses.
data_preprocessing = None
LAMBDA_GP = 10


train_data_reader = SequenceBodyReader('splitted_skeleton\\train_map.csv', sequence_length, dataset,skip_frame=0,data_preprocessing=data_preprocessing,random_sequence=False)
# setting up the model
minibatch_size = 16
lr_init = 5e-5

d_lr = lr_init
g_lr = lr_init    
epoch = 0

lossGenerator = []
lossCritic = []
lossDiscriminator = []

z_rand_params = {'low':-0.1, 'high':0.1}
def generate_random(params, shape):
    return np.random.uniform(params['low'], params['high'], size=shape)


generator = Generator(z_size,inputs_depth, num_layers, output_size)
discriminator = Discriminator(input_discriminator_size, inputs_depth)
critic = Critic(input_discriminator_size, inputs_depth)

opt_disc = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9))
opt_gen = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=d_lr, betas=(0.0, 0.9))
criterion = nn.BCELoss()




for epoch in range(max_epochs):
    train_data_reader.reset()
    idx = 0
    while(train_data_reader.has_more()):
     
     idx += 1
     input_data, _, current_batch_size, _, _ = train_data_reader.next_minibatch(minibatch_size)
     input_data = input_data / np.linalg.norm(input_data)
     real_poses = torch.from_numpy(input_data)
     input_data = input_data.reshape(-1, sequence_length, input_size)
     prior_poses = real_poses[:, :10, :]
     future_poses = real_poses[:, 10:, :]
     real_poses = real_poses.reshape(-1, input_discriminator_size)
     
     z_data = generate_random(z_rand_params, shape=[current_batch_size, z_size])
     z_data = torch.from_numpy(z_data)
     z_data = z_data.to(torch.float32)
    
     fake_poses = generator(z_data)
     fake_poses = torch.cat((prior_poses, fake_poses), axis=1)
     print(fake_poses.shape)
     visualizer.draw_to_file(fake_poses)
     fake_poses = fake_poses.reshape(-1, input_discriminator_size)
     
     
    #Train critic
     critic_real = critic(real_poses).view(-1) 
     critic_fake = critic(fake_poses).view(-1)
     gp = gradient_penalty(critic, real_poses, fake_poses)
     lossC = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp)
     critic.zero_grad()
     lossC.backward(retain_graph=True)
     opt_critic.step() 
     
    #Train discriminator
     disc_real = discriminator(real_poses).view(-1)
     if(idx==10):
      accuracy_real_poses = torch.mean(disc_real)*100
      print("accuracy_real_poses: %f% ", accuracy_real_poses.item())
     lossD_real = criterion(disc_real, torch.ones_like(disc_real))
     disc_fake = discriminator(fake_poses).view(-1)
     if(idx==10):
      accuracy_fake_poses = torch.mean(disc_fake)*100
      print("accuracy_fake_poses: %f%", accuracy_fake_poses.item()) 
     lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
     lossD = (lossD_real + lossD_fake) / 2
     discriminator.zero_grad()
     lossD.backward(retain_graph=True)
     opt_disc.step()
    
    
     #Train generator
     output = critic(fake_poses).view(-1)
     lossG = -torch.mean(output)
     generator.zero_grad()
     lossG.backward()
     opt_gen.step()
     
    
        
    
    lossGenerator.append(lossG.item())
    lossCritic.append(lossC.item())
    lossDiscriminator.append(lossD.item())
    

    print(f"Epoch [{epoch}/{max_epochs}] LossC: {lossC:.4f} lossD: {lossD:.4f} loss G: {lossG:.4f}")
 


plt.plot(lossCritic, '-b', label='lossGenerator')
plt.plot(lossDiscriminator, '-r', label='lossDiscriminator')
plt.plot(lossGenerator, '-g', label='lossGenerator')
plt.title("Losses")
plt.savefig("Losses1.png")

plt.show()