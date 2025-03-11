from boost_corr.xpcs_aps_8idi.rigaku_3M_handler import Rigaku3MDataset

fname = '/home/8ididata/Rigaku3M_Test_Data/SparsificationData2/Div1_000001.bin.000'
dset = Rigaku3MDataset(fname, batch_size=1000, device='cuda:0')
print(dset.det_size)
print(dset.frame_num)
# plt.imshow(dset.get_scattering())
# plt.show()

for n in range(100):
    x = dset.__getbatch__(n)
    print(n, x.shape, x.device)

