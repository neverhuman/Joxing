"""H28-only sampling profile: original monitor/100ms gates, 1ms requested poll.

Fast, small fake children can leave the terminal-mm phase between 20ms polls.
No synthetic delay or forced exit is inserted; missing positive evidence fails.
"""

from functools import partial
from unittest.mock import patch

import trial_rss_watchdog as original

if __name__ == "__main__":
    with patch.object(original, "monitor", partial(original.monitor, poll_s=0.001)):
        original.main()
