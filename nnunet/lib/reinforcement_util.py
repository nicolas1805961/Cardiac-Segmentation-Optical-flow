def convert_state(self, state):
        angle = state[0].item()
        tx = state[1].item()
        ty = state[2].item()
        angle = angle * (156 / 4)
        tx = tx * (83 / 4)
        ty = ty * (57 / 4)
        return torch.tensor([angle, tx, ty])
    
    def get_reinforcement_validation_images(self):
        embedding = torch.nn.Embedding(4, 3)
        embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=self.device)
        self.policy_net.eval()
        with torch.no_grad():
            data = next(iter(self.validation_random_dataloader))
            x, y, parameters, center = data['x'], data['y'], data['parameters'].squeeze(), data['center'].squeeze()

            #fig, ax = plt.subplots(2, 1)
            #print(center)
            #print(unstandardize_parameters(parameters[0]))
            #ax[0].imshow(x[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(torch.argmax(y, dim=1)[0].cpu(), cmap='gray')
            #plt.show()

            print(unstandardize_parameters(parameters))
            print('*************************')
            y = torch.argmax(y, dim=1, keepdim=True)
            current_image = torch.clone(x)
            Tt = torch.zeros((self.number_of_steps, 3), dtype=torch.float64)
            j = 0
            while j < self.number_of_steps - 1:
                out = self.policy_net(current_image)
                action = torch.argmax(out, dim=1)
                Tt[j + 1, :] = self.q_value.take_action(action, Tt[j, :])
                print(self.convert_state(Tt[j + 1]))
                if torch.unique(Tt[:j + 2], dim=0).size(0) == torch.unique(Tt[:j + 1], dim=0).size(0):
                    break
                current_image = self.get_new_image(x, self.convert_state(Tt[j + 1]), center, InterpolationMode.BILINEAR)
                current_image_y = self.get_new_image(y, self.convert_state(Tt[j + 1]), center, InterpolationMode.NEAREST)
                j += 1
            print(f'Number of steps used: {j}')

            x = cv.normalize(x.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            y = embedding(y.squeeze())
            current_image = cv.normalize(current_image.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            current_image_y = embedding(current_image_y.squeeze())
            out_dict = {'pred': current_image, 'x': x, 'y': y.cpu().numpy().astype(np.uint8), 'pred_y': current_image_y.cpu().numpy().astype(np.uint8)}
            return out_dict
    
    def reinforcement_validation_loop(self):
        self.policy_net.eval()
        with torch.no_grad():
            correct = 0
            for data in tqdm(self.val_dataloader_subset, desc='Validation iteration: ', position=2):
                x, q = data['x'], data['q_values'].squeeze()
                out = self.policy_net(x).squeeze()
                #print(q)
                #print(out)
                #print('***********************')
                correct += int(torch.argmax(out) == torch.argmax(q))
            correct = correct / len(self.val_dataloader_subset)
            return correct
    
    def get_new_image(self, x, state, center, interpolation):
        angle = state[0].item()
        tx = state[1].item()
        ty = state[2].item()
        out = affine(x, angle=angle, translate=[tx, ty], scale=1, shear=0, interpolation=interpolation, center=center.tolist()[::-1])
        return out


    def reinforcement_test(self):
        q_value = Q_value(0.9, 0.5, 10, 6, 1, 1)
        number_of_steps = 100
        self.policy_net.eval()
        with torch.no_grad():
            for data in tqdm(self.val_dataloader_subset, desc='Validation iteration: ', position=2):
                x, q, parameters = data['x'], data['q_values'], data['parameters']
                print(parameters)
                print('*************************')
                current_image = torch.clone(x)
                Tt = torch.zeros((number_of_steps, 3), dtype=torch.int16)
                j = 0
                while j < number_of_steps - 1:
                    out = self.policy_net(current_image)
                    action = torch.argmax(out, dim=1)
                    Tt[j + 1, :] = q_value.take_action(action, Tt[j, :])
                    print(Tt[j + 1, :])
                    if torch.unique(Tt[:j + 2], dim=0).size(0) == torch.unique(Tt[:j + 1], dim=0).size(0):
                        break
                    current_image = self.get_new_image(x, Tt[j + 1])
                    j += 1