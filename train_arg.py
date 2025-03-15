from robustness import attacker
def add_args_to_parser(arg_list, parser):
    """
    Adds arguments from one of the argument lists above to a passed-in
    arparse.ArgumentParser object. Formats helpstrings according to the
    defaults, but does NOT set the actual argparse defaults (*important*).

    Args:
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        parser (argparse.ArgumentParser) : An ArgumentParser object to which the
            arguments will be added

    Returns:
        The original parser, now with the arguments added in.
    """
    for arg_name, arg_type, arg_help, arg_default in arg_list:
        has_choices = (type(arg_type) == list) 
        kwargs = {
            'type': type(arg_type[0]) if has_choices else arg_type,
            'default': arg_default,
            'help': f"{arg_help} (default: {arg_default})"
        }
        if has_choices: kwargs['choices'] = arg_type
        parser.add_argument(f'--{arg_name}', **kwargs)
    return parser

SAM_ARGS = [
    ['minimizer', ['SAM','ASAM'], 'the type of sam', 'SAM'],
    ['rho', float, 'the rho for ASAM', 0.2],
    ['eta', float, 'eta for ASAM', 0.0],
]

TRAINING_ARGS = [
    ['epoch', int, 'number of epochs to train for', 200],
    # ['arch',['resnet50', 'densenet121','mobilenetv2_100','vgg16','vgg16_bn','inception_v3','xception41','vit_base_patch16_224','inception_resnet_v2'],"",'resnet50'],
    ['arch',str,"",'resnet50'],
    ['batch_size', int, 'number of epochs to train for', 128],
    ['data_dir', str, 'image dir same with loaded image dir', "/private/dataset/imagenette"],
    ['dataset', ['cifar','imagenette'], 'start with random noise instead of pgd step', 'imagenette'],
]

EVAL_ATTACK_ARGS = [
    ['trans_attack_steps', int, 'number of steps for iterative attack', 50],
    ['trans_eps', float , 'adversarial perturbation budget', 8/255],
    ['trans_attack_lr', float, 'step size for iterative attack', 2/255],
    ['trans_random_start', [0, 1], 'start with random noise instead of pgd step', 1],
    ['attack_type',str,'eval attack type','mifgsm'],
    ['defence_type', ['none', 'jpeg', 'fd_jpeg', 'bit_depth', 'at'], 'choose a defence', 'none'],
]


PGD_ARGS = [
    ['need_attack', bool, 'eps mult. sched (same format as LR)', True],
    ['attack_steps', int, 'number of steps for PGD attack', 7],
    ['constraint', list(attacker.STEPS.keys()), 'adv constraint', 2],
    ['eps', float , 'adversarial perturbation budget', 1],
    ['attack_lr', float, 'step size for PGD (about 1/4 of eps)', 0.4],
    ['targeted', [0, 1], 'if 1 (0) is (is not) a target attack', 0],
    ['use_best', [0, 1], 'if 1 (0) use best (final) PGD step as example', 1],
    ['random_restarts', int, 'number of random PGD restarts for eval', 0],
    ['random_start', [0, 1], 'start with random noise instead of pgd step', 1],
    ['custom_eps_multiplier', str, 'eps mult. sched (same format as LR)', None],
    ['eps_for_division', float, 'stop to divide 0', 1e-10],
    ['fuse_weight', float, 'when use weighted loss schema, the ratio of clean loss fuse', 0.3],
    ['loss_schema', ['averaged', 'weighted'], 'how to fuse loss', 'averaged'],
    ['saved_dir', str, 'where to save the model', ''],
]
