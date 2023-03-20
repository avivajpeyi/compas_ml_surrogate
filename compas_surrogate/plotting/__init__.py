from compas_surrogate.logger import logger


def safe_savefig(fig, fname, *args, **kwargs):
    """Save the figure to a file."""
    try:
        fig.savefig(fname, *args, **kwargs)
    except Exception as e:
        logger.error(f"Could not save figure to {fname}: {e}")
