import logging

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Hello from py-execution-engine!")


if __name__ == "__main__":
    main()
