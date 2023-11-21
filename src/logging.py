import typing as t

DELIMITER = ', '

def log_metrics_to_file(metrics: t.Dict[str, float], log_path: str, write_mode: str = 'w', epoch: t.Optional[int] = None) -> None:    
    with open(log_path, write_mode) as f:
        if write_mode == 'w':
            prefix = 'epoch, ' if epoch is not None else ''
            f.write(f'{prefix}{str.join(DELIMITER, metrics.keys())}\n')

        prefix = f'{epoch}, ' if epoch is not None else ''
        metrics_values_str = map(lambda x: str(x), metrics.values())
        f.write(f'{prefix}{str.join(DELIMITER, metrics_values_str)}\n')
