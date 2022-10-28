# import matplotlib.pyplot as plt

# def printImage(image, label):
#     plt.imshow(image)
#     if label < K:
#         print(f'Label: idx({label}) class name: {wnID_to_names[idx_to_class[label]]}')
#     else:
#         subclass_names = []
#         for subclass in idx_to_class[label]:
#             subclass_names.append(wnID_to_names[subclass])
#         print(f'Label: idx({label}) class name: {subclass_names}')

# if __name__ == '__main__': 
#     # test the vagued data 
#     # # train non-vague 
#     indexes = [i for i in range(len(train_ds)) if train_ds[i][1] == C[0][0]]
#     print(f"Total number of non-vague training samples per subclass in \'{vague_classes[0]}\': {len(indexes)}")
#     idx = indexes[2]
#     printImage(train_ds[idx][0], train_ds[idx][1])

#     # # train vague 
#     vague_indexes = [i for i in range(len(train_ds)) if train_ds[i][1] == K]
#     print(f"Total number of vague training samples in \'{vague_classes[0]}\': {len(vague_indexes)}")
#     vague_idx = vague_indexes[8]
#     printImage(train_ds[vague_idx][0], train_ds[vague_idx][1])

#     # # Test non-vague 
#     indexes = [i for i in range(len(test_ds)) if test_ds[i][1] == C[0][1]]
#     print("Total number of non-vague test samples per subclass in \'{}\': {}".format(vague_classes[0], len(indexes)))
#     idx = indexes[2]
#     printImage(test_ds[idx][0], test_ds[idx][1])

#     # # Test vague
#     vague_indexes = [i for i in range(len(test_ds)) if test_ds[i][1] == K]
#     print("Total number of vague training samples in \'{}\': {}".format(vague_classes[0], len(vague_indexes)))

#     vague_idx = vague_indexes[8]
#     printImage(test_ds[vague_idx][0], test_ds[vague_idx][1])

