
echo "Finding Dependencies <viz toolkit>,<sfm pkg>,<calib3d pkg>,<cvcore>,<xfeatures2d>,<features2d>,<highgui>,<ceres solver>"

r=`tput setaf 1`
rst=`tput sgr0`
g=`tput setaf 2`

#find /lib* /usr/local/include/ /usr/include/ -name '*viz.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*viz.hpp*' | wc -l)
pkgs="viz.hpp"
echo;
if [ $lines -eq 0 ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*sfm.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*sfm.hpp*' | wc -l)
pkgs="sfm.hpp"
echo;
if [ $lines -eq 0 ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*calib3d.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*calib3d.hpp*' | wc -l)
pkgs="calib3d.hpp"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*coreny.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*coreny.hpp*' | wc -l)
pkgs="core.hpp"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*xfeatures2d.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*xfeatures2d.hpp*' | wc -l)
pkgs="xfeatures2d.hpp"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;

#find /lib* /usr/local/include/ /usr/include/ -name '*features2d.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*features2d.hpp*' | wc -l)
pkgs="features2d.hpp"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*highgui.hpp*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*highgui.hpp*' | wc -l)
pkgs="highgui.hpp"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi
echo;
#find /lib* /usr/local/include/ /usr/include/ -name '*ceres*'
lines=$(find /lib* /usr/local/include/ /usr/include/ -name '*ceres*' | wc -l)
pkgs="ceres"
echo;
if [ "$lines" = "0" ]; then
  echo "$r---x $pkgs not found in include path$rst"
else
  echo "$g---> $pkgs FOUND w/ $lines copy$rst"
fi


