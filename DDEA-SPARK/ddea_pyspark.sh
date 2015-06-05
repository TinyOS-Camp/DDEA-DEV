#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Figure out where Spark is installed
FWDIR="$(cd "`dirname "$0"`"/..; pwd)"
DDEA_BIN="/adsc/DDEA_PROTO/bin"

# Export this as SPARK_HOME
export SPARK_HOME="$FWDIR"

source "$FWDIR/bin/utils.sh"

source "$FWDIR"/bin/load-spark-env.sh

function usage() {
  echo "Usage: ./bin/pyspark [options]" 1>&2
  "$FWDIR"/bin/spark-submit --help 2>&1 | grep -v Usage 1>&2
  exit 0
}

if [[ "$@" = *--help ]] || [[ "$@" = *-h ]]; then
  usage
fi

# Exit if the user hasn't compiled Spark
if [ ! -f "$FWDIR/RELEASE" ]; then
  # Exit if the user hasn't compiled Spark
  ls "$FWDIR"/assembly/target/scala-$SPARK_SCALA_VERSION/spark-assembly*hadoop*.jar >& /dev/null
  if [[ $? != 0 ]]; then
    echo "Failed to find Spark assembly in $FWDIR/assembly/target" 1>&2
    echo "You need to build Spark before running this program" 1>&2
    exit 1
  fi
fi

# In Spark <= 1.1, setting IPYTHON=1 would cause the driver to be launched using the `ipython`
# executable, while the worker would still be launched using PYSPARK_PYTHON.
#
# In Spark 1.2, we removed the documentation of the IPYTHON and IPYTHON_OPTS variables and added
# PYSPARK_DRIVER_PYTHON and PYSPARK_DRIVER_PYTHON_OPTS to allow IPython to be used for the driver.
# Now, users can simply set PYSPARK_DRIVER_PYTHON=ipython to use IPython and set
# PYSPARK_DRIVER_PYTHON_OPTS to pass options when starting the Python driver
# (e.g. PYSPARK_DRIVER_PYTHON_OPTS='notebook').  This supports full customization of the IPython
# and executor Python executables.
#
# For backwards-compatibility, we retain the old IPYTHON and IPYTHON_OPTS variables.

# Determine the Python executable to use if PYSPARK_PYTHON or PYSPARK_DRIVER_PYTHON isn't set:
if hash python2.7 2>/dev/null; then
  # Attempt to use Python 2.7, if installed:
  DEFAULT_PYTHON="$DDEA_BIN/python2.7"
else
  DEFAULT_PYTHON="$DDEA_BIN/python"
fi

# Determine the Python executable to use for the driver:
if [[ -n "$IPYTHON_OPTS" || "$IPYTHON" == "1" ]]; then
  # If IPython options are specified, assume user wants to run IPython
  # (for backwards-compatibility)
  PYSPARK_DRIVER_PYTHON_OPTS="$PYSPARK_DRIVER_PYTHON_OPTS $IPYTHON_OPTS"
  PYSPARK_DRIVER_PYTHON="ipython"
elif [[ -z "$PYSPARK_DRIVER_PYTHON" ]]; then
  PYSPARK_DRIVER_PYTHON="${PYSPARK_PYTHON:-"$DEFAULT_PYTHON"}"
fi

# Determine the Python executable to use for the executors:
if [[ -z "$PYSPARK_PYTHON" ]]; then
  if [[ $PYSPARK_DRIVER_PYTHON == *ipython* && $DEFAULT_PYTHON != "python2.7" ]]; then
    echo "IPython requires Python 2.7+; please install python2.7 or set PYSPARK_PYTHON" 1>&2
    exit 1
  else
    PYSPARK_PYTHON="$DEFAULT_PYTHON"
  fi
fi
export PYSPARK_PYTHON

# Add the PySpark classes to the Python path:
export PYTHONPATH="$SPARK_HOME/python/:$PYTHONPATH"
export PYTHONPATH="$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH"

# Load the PySpark shell.py script when ./pyspark is used interactively:
export OLD_PYTHONSTARTUP="$PYTHONSTARTUP"
export PYTHONSTARTUP="$FWDIR/python/pyspark/shell.py"

# Build up arguments list manually to preserve quotes and backslashes.
# We export Spark submit arguments as an environment variable because shell.py must run as a
# PYTHONSTARTUP script, which does not take in arguments. This is required for IPython notebooks.
SUBMIT_USAGE_FUNCTION=usage
gatherSparkSubmitOpts "$@"
PYSPARK_SUBMIT_ARGS=""
whitespace="[[:space:]]"
for i in "${SUBMISSION_OPTS[@]}"; do
  if [[ $i =~ \" ]]; then i=$(echo $i | sed 's/\"/\\\"/g'); fi
  if [[ $i =~ $whitespace ]]; then i=\"$i\"; fi
  PYSPARK_SUBMIT_ARGS="$PYSPARK_SUBMIT_ARGS $i"
done
export PYSPARK_SUBMIT_ARGS

# For pyspark tests
if [[ -n "$SPARK_TESTING" ]]; then
  unset YARN_CONF_DIR
  unset HADOOP_CONF_DIR
  if [[ -n "$PYSPARK_DOC_TEST" ]]; then
    exec "$PYSPARK_DRIVER_PYTHON" -m doctest $1
  else
    exec "$PYSPARK_DRIVER_PYTHON" $1
  fi
  exit
fi

# If a python file is provided, directly run spark-submit.
if [[ "$1" =~ \.py$ ]]; then
  echo -e "\nWARNING: Running python applications through ./bin/pyspark is deprecated as of Spark 1.0." 1>&2
  echo -e "Use ./bin/spark-submit <python file>\n" 1>&2
  primary="$1"
  shift
  gatherSparkSubmitOpts "$@"
  exec "$FWDIR"/bin/spark-submit "${SUBMISSION_OPTS[@]}" "$primary" "${APPLICATION_OPTS[@]}"
else
  exec "$PYSPARK_DRIVER_PYTHON" $PYSPARK_DRIVER_PYTHON_OPTS
fi
