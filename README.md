**Exception Handling**
*mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64') or (have 'x86_64' need 'arm64')*
1. find error package name e.g. *package_name*
2. pip uninstall *package_name*
3. arch -x86_64 pip install *package_name*
3. arch -arm64 pip install *package_name*
4. pip freeze