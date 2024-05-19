path = "/content/drive/My Drive/2022 AI/GAN Model/Saved Models"

def showOutputsGrid(generator, device):
	generator.eval()
	nrow = 8
	ncol = 8
	noise = 100
	num = nrow * ncol
	noise = torch.randn(num, noise, 1, 1, device=device)
	imgList = []

	ckpts = glob.glob(path+'/generatorCheckpoint*.pt')
	ckpts.append(path+'/generator.pt')

	for ckpt in ckpts:
		generator.load_state_dict(torch.load(ckpt))
		with torch.no_grad():
			output = generator(noise).to('cpu')

		grid = vutils.make_grid(output, nrow=ncol, pad_value=2)
		plt.axis("off")
		plt.imshow(np.transpose(grid, (1,2,0)))
		plt.show()

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	generator = Generators(1).to(device)
	showOutputsGrid(generator, device)

if __name__ == '__main__':
    main()
