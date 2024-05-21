import torch
import time


def main(size=1024, num_channels=1024 * 1024, num_tries=1, device='cuda:0'):
    v1 = torch.rand(size, num_channels, device=device)
    v2 = torch.rand(size, num_channels, device=device)

    torch.linalg.vecdot(v1, v2, dim=1)
    t0 = time.perf_counter()
    for n in range(num_tries):
        result1 = torch.linalg.vecdot(v1, v2, dim=1)
    t1 = time.perf_counter()
    print('method 1:', t1 - t0)

    torch.sum(v1 * v2, dim=1)
    t0 = time.perf_counter()
    for n in range(num_tries):
        result2 = torch.sum(v1 * v2, dim=1)
    t1 = time.perf_counter()
    print('method 2:', t1 - t0)

    print(torch.allclose(result1, result2))

if __name__ == '__main__':
    main()