import argparse

def preprocess_args(available_datasets: list):
    parser = argparse.ArgumentParser(description='Preprocess command line arguments.')
    parser.add_argument('-a', '--all', action='store_true', help='Preprocess all available original datasets')
    parser.add_argument('-d', '--datasets', nargs='+', help='Specify original datasets name')
    parser.add_argument('-l', '--list-all', action='store_true', help='List all available original datasets')

    args = parser.parse_args()

    if args.all:
        return available_datasets

    if args.list_all:
        print(f"{len(available_datasets)} available original datasets:")
        for dataset in available_datasets:
            print(dataset)
        exit()
    
    has_args = False

    datasets = []

    if args.datasets:
        has_args = True
        for dataset in args.datasets:
            if dataset not in available_datasets:
                print(f"Original dataset '{dataset}' not found. Using -l to list all available datasets.")
                exit()
            datasets.append(dataset)
    
    if not has_args:
        # 如果用户没有提供任何选项和参数，则打印帮助信息
        parser.print_help()
        exit()
    
    if len(datasets) == 0:
        datasets = available_datasets
        
    return datasets


def bench_args(available_datasets: list, available_methods: list):
    # 构建命令行参数解析器
    parser = argparse.ArgumentParser(description='Benchmark command line arguments.')
    parser.add_argument('-a', '--all', action='store_true', help='Run all available datasets and methods')
    parser.add_argument('-d', '--datasets', nargs='+', help='Specify datasets names')
    parser.add_argument('-m', '--methods', nargs='+', help='Specify methods names')
    parser.add_argument('-l', '--list-all', action='store_true', help='List all available datasets and methods')
    parser.add_argument('-ld', '--list-datasets', action='store_true', help='List all available datasets')
    parser.add_argument('-lm', '--list-methods', action='store_true', help='List all available methods')

    # 解析命令行参数
    args = parser.parse_args()

    if args.all:
        datasets = available_datasets
        methods = available_methods
        return datasets, methods
    
    # 输出所有可用的数据集和降维方法
    if args.list_all:
        print(f"{len(available_datasets)} available datasets:")
        for dataset in available_datasets:
            print(dataset)
        print(f"{len(available_methods)} available methods:")
        for method in available_methods:
            print(method)
        exit()
    
    # 列出所有可用的数据集
    if args.list_datasets:
        print(f"{len(available_datasets)} available datasets:")
        for dataset in available_datasets:
            print(dataset)
        exit()

    # 列出所有可用的降维方法
    if args.list_methods:
        print(f"{len(available_methods)} available methods:")
        for method in available_methods:
            print(method)
        exit()

    has_args = False

    datasets = []
    methods = []

    # 输出指定的数据集和降维方法
    if args.datasets:
        has_args = True
        for dataset in args.datasets:
            if dataset not in available_datasets:
                print(f"Dataset '{dataset}' not found. Using -l or -ld to list all available datasets.")
                exit()
            datasets.append(dataset)
    if args.methods:
        has_args = True
        for method in args.methods:
            if method not in available_methods:
                print(f"Method '{method}' not found. Using -l or -lm to list all available methods.")
                exit()
            methods.append(method)
    
    if not has_args:
        # 如果用户没有提供任何选项和参数，则打印帮助信息
        parser.print_help()
        exit()

    if len(datasets) == 0:
        datasets = available_datasets
    if len(methods) == 0:
        methods = available_methods

    return datasets, methods

