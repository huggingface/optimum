# Register commands in the Optimum CLI from a subpackage

It is possible to register a command in the Optimum CLI, either as a command or a subcommand of an already existing command.

Steps to follow:

1. Create a command as a subclass of `optimum.commands.BaseOptimumCLICommand`.
2. Create a Python file under `optimum/commands/register/`, and define a `REGISTER_COMMANDS` list variable there.
3. Fill the `REGISTER_COMMANDS` as folows:

```python
# CustomCommand1 and CustomCommand2 could also be defined in this file actually.
from ..my_custom_commands import CustomCommand1, CustomCommand2
from ..export import ExportCommand

REGISTER_COMMANDS = [
  # CustomCommand1 will be registered as a subcommand of the root Optimum CLI. 
  CustomCommand1, 
  # CustomCommand2 will be registered as a subcommand of the `optimum-cli export` command. 
  (CustomCommand2, ExportCommand) # CustomCommand2 will be registered
]
```
