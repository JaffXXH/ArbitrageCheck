
#------------------------------------------------------------------------
# version_manager.py

import os
import re
from datetime import datetime
from version_config import VERSION_CONFIG

class VersionManager:
    def __init__(self, config=None):
        self.config = config or VERSION_CONFIG
        self.version_file = self.config["version_file_path"]
    
    def parse_version(self, version_string):
        """Parse version string into components"""
        match = re.match(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', version_string)
        if match:
            return tuple(int(x) for x in match.groups())
        raise ValueError(f"Invalid version format: {version_string}")
    
    def format_version(self, version_tuple):
        """Format version tuple into string"""
        return ".".join(str(x) for x in version_tuple)
    
    def increment_version(self, version_string, increment_type="build"):
        """
        Increment version based on type:
        - major: 1.0.0.0 -> 2.0.0.0
        - minor: 1.0.0.0 -> 1.1.0.0
        - patch: 1.0.0.0 -> 1.0.1.0
        - build: 1.0.0.0 -> 1.0.0.1 (default)
        """
        major, minor, patch, build = self.parse_version(version_string)
        
        if increment_type == "major":
            return self.format_version((major + 1, 0, 0, 0))
        elif increment_type == "minor":
            return self.format_version((major, minor + 1, 0, 0))
        elif increment_type == "patch":
            return self.format_version((major, minor, patch + 1, 0))
        elif increment_type == "build":
            return self.format_version((major, minor, patch, build + 1))
        else:
            raise ValueError(f"Unknown increment type: {increment_type}")
    def get_exe_version(self, exe_path):
      """Get version information from executable file"""
      try:
          import win32api
          info = win32api.GetFileVersionInfo(exe_path, "\\")
          ms = info['FileVersionMS']
          ls = info['FileVersionLS']
          version = f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
          return version
      except Exception as e:
          print(f"Error reading version: {e}")
          return None
      
    def get_current_version(self):
        """Get current version from config or version file"""
        try:
            # Try to read from existing version file first
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract version from file
                    match = re.search(r"filevers=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", content)
                    if match:
                        return self.format_version(tuple(int(x) for x in match.groups()))
        except Exception as e:
            print(f"Warning: Could not read version from file: {e}")
        
        # Fallback to config
        return self.config["current_version"]
    
    def update_version(self, new_version=None, increment_type="build"):
        """Update to specific version or auto-increment"""
        current_version = self.get_current_version()
        
        if new_version:
            # Use specified version
            next_version = new_version
        elif self.config.get("auto_increment", True):
            # Auto-increment
            next_version = self.increment_version(current_version, increment_type)
        else:
            # Keep current version
            next_version = current_version
        
        print(f"Updating version: {current_version} -> {next_version}")
        return next_version
    
    def generate_version_file_content(self, version):
        """Generate the version.txt file content"""
        major, minor, patch, build = self.parse_version(version)
        
        template = f"""# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    # filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
    # Set not needed items to zero 0.
    filevers=({major}, {minor}, {patch}, {build}),
    prodvers=({major}, {minor}, {patch}, {build}),
    # Contains a bitmask that specifies the valid bits 'flags'r
    mask=0x3f,
    # Contains a bitmask that specifies the Boolean attributes of the file.
    flags=0x0,
    # The operating system for which this file was designed.
    # 0x4 - NT and there is no need to change it.
    OS=0x40004,
    # The general type of file.
    # 0x1 - the file is an application.
    fileType=0x1,
    # The function of the file.
    # 0x0 - the function is not defined for this fileType
    subtype=0x0,
    # Creation date and time stamp.
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'{self.config["company_name"]}'),
        StringStruct(u'FileDescription', u'{self.config["file_description"]}'),
        StringStruct(u'FileVersion', u'{version}'),
        StringStruct(u'InternalName', u'{self.config["internal_name"]}'),
        StringStruct(u'LegalCopyright', u'{self.config["legal_copyright"]}'),
        StringStruct(u'OriginalFilename', u'{self.config["original_filename"]}'),
        StringStruct(u'ProductName', u'{self.config["product_name"]}'),
        StringStruct(u'ProductVersion', u'{version}')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [0x409, 1200])])
  ]
)
"""
        return template
    
    def create_version_file(self, version=None, increment_type="build"):
        """Create or update the version.txt file"""
        if version is None:
            version = self.update_version(increment_type=increment_type)
        
        content = self.generate_version_file_content(version)
        
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Version file created/updated: {self.version_file}")
            print(f"✓ Version set to: {version}")
            
            # Also update the config
            self.config["current_version"] = version
            
            return version
        except Exception as e:
            print(f"✗ Error creating version file: {e}")
            return None
    
    def display_version_info(self):
        """Display current version information"""
        current_version = self.get_current_version()
        print(f"Current Version: {current_version}")
        print(f"Version File: {self.version_file}")
        print(f"Auto-increment: {self.config.get('auto_increment', True)}")

# Convenience functions
def create_version_file(version=None, increment_type="build"):
    """Convenience function to create version file"""
    manager = VersionManager()
    return manager.create_version_file(version, increment_type)

def get_current_version():
    """Convenience function to get current version"""
    manager = VersionManager()
    return manager.get_current_version()

if __name__ == "__main__":
    # Test the version manager
    manager = VersionManager()
    manager.display_version_info()
    
    # Create initial version file
    manager.create_version_file()
