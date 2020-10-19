(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{106:function(e,n,t){"use strict";t.d(n,"a",(function(){return d})),t.d(n,"b",(function(){return m}));var r=t(0),a=t.n(r);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function c(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?c(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):c(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)t=o[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var u=a.a.createContext({}),s=function(e){var n=a.a.useContext(u),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},d=function(e){var n=s(e.components);return a.a.createElement(u.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return a.a.createElement(a.a.Fragment,{},n)}},b=a.a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,o=e.originalType,c=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),d=s(t),b=r,m=d["".concat(c,".").concat(b)]||d[b]||p[b]||o;return t?a.a.createElement(m,i(i({ref:n},u),{},{components:t})):a.a.createElement(m,i({ref:n},u))}));function m(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var o=t.length,c=new Array(o);c[0]=b;var i={};for(var l in n)hasOwnProperty.call(n,l)&&(i[l]=n[l]);i.originalType=e,i.mdxType="string"==typeof e?e:r,c[1]=i;for(var u=2;u<o;u++)c[u]=t[u];return a.a.createElement.apply(null,c)}return a.a.createElement.apply(null,t)}b.displayName="MDXCreateElement"},69:function(e,n,t){"use strict";t.r(n),t.d(n,"frontMatter",(function(){return c})),t.d(n,"metadata",(function(){return i})),t.d(n,"rightToc",(function(){return l})),t.d(n,"default",(function(){return s}));var r=t(2),a=t(6),o=(t(0),t(106)),c={title:"Azure ML Containers"},i={unversionedId:"docker-build",id:"docker-build",isDocsHomePage:!1,title:"Azure ML Containers",description:"In this post we explain how Azure ML builds the containers used to run your code.",source:"@site/docs/docker-build.md",slug:"/docker-build",permalink:"/AzureML-CheatSheet/docs/docker-build",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/docs/docker-build.md",version:"current",sidebar:"mainSidebar",previous:{title:"Distributed GPU Training",permalink:"/AzureML-CheatSheet/docs/distributed-training"}},l=[{value:"Dockerfile",id:"dockerfile",children:[]}],u={rightToc:l};function s(e){var n=e.components,t=Object(a.a)(e,["components"]);return Object(o.b)("wrapper",Object(r.a)({},u,t,{components:n,mdxType:"MDXLayout"}),Object(o.b)("p",null,"In this post we explain how Azure ML builds the containers used to run your code."),Object(o.b)("h2",{id:"dockerfile"},"Dockerfile"),Object(o.b)("p",null,"Each job in Azure ML runs with an associated ",Object(o.b)("inlineCode",{parentName:"p"},"Environment"),". In practice, each environment\ncorresponds to a Docker image."),Object(o.b)("p",null,"There are numerous ways to define an environment - from specifying a set of required Python packages\nthrough to directly providing a custom Docker image. In each case the contents of the associated\ndockerfile are available directly from the environment object."),Object(o.b)("p",null,"For more background: ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"environment"}),"Environment")),Object(o.b)("h4",{id:"example"},"Example"),Object(o.b)("p",null,"Suppose you create an environment - in this example we will work with Conda:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-yml",metastring:'title="env.yml"',title:'"env.yml"'}),"name: pytorch\nchannels:\n    - defaults\n    - pytorch\ndependencies:\n    - python=3.7\n    - pytorch\n    - torchvision\n")),Object(o.b)("p",null,"We can create and register this as an ",Object(o.b)("inlineCode",{parentName:"p"},"Environment")," in our workspace ",Object(o.b)("inlineCode",{parentName:"p"},"ws")," as follows:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"from azureml.core import Environment\nenv = Environment.from_conda_specification('pytorch', 'env.yml')\nenv.register(ws)\n")),Object(o.b)("p",null,"In order to consume this environment in a remote run, Azure ML builds a docker image\nthat creates the corresponding python environment."),Object(o.b)("p",null,"The dockerfile used to build this image is available directly from the environment object."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"details = env.get_image_details(ws)\nprint(details['ingredients']['dockerfile'])\n")),Object(o.b)("p",null,"Let's take a look:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-docker",metastring:'title="Dockerfile" {1,7-12}',title:'"Dockerfile"',"{1,7-12}":!0}),'FROM mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20200821.v1@sha256:8cee6f674276dddb23068d2710da7f7f95b119412cc482675ac79ba45a4acf99\nUSER root\nRUN mkdir -p $HOME/.cache\nWORKDIR /\nCOPY azureml-environment-setup/99brokenproxy /etc/apt/apt.conf.d/\nRUN if dpkg --compare-versions `conda --version | grep -oE \'[^ ]+$\'` lt 4.4.11; then conda install conda==4.4.11; fi\nCOPY azureml-environment-setup/mutated_conda_dependencies.yml azureml-environment-setup/mutated_conda_dependencies.yml\nRUN ldconfig /usr/local/cuda/lib64/stubs && conda env create -p /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6 -\nf azureml-environment-setup/mutated_conda_dependencies.yml && rm -rf "$HOME/.cache/pip" && conda clean -aqy && CONDA_ROO\nT_DIR=$(conda info --root) && rm -rf "$CONDA_ROOT_DIR/pkgs" && find "$CONDA_ROOT_DIR" -type d -name __pycache__ -exec rm\n -rf {} + && ldconfig\n# AzureML Conda environment name: azureml_7459a71437df47401c6a369f49fbbdb6\nENV PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/bin:$PATH\nENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6\nENV LD_LIBRARY_PATH /azureml-envs/azureml_7459a71437df47401c6a369f49fbbdb6/lib:$LD_LIBRARY_PATH\nCOPY azureml-environment-setup/spark_cache.py azureml-environment-setup/log4j.properties /azureml-environment-setup/\nRUN if [ $SPARK_HOME ]; then /bin/bash -c \'$SPARK_HOME/bin/spark-submit  /azureml-environment-setup/spark_cache.py\'; fi\nENV AZUREML_ENVIRONMENT_IMAGE True\nCMD ["bash"]\n')),Object(o.b)("p",null,"Notice:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"The base image here is a standard image maintained by Azure ML. Dockerfiles for all base images are available on\ngithub: ",Object(o.b)("a",Object(r.a)({parentName:"li"},{href:"https://github.com/Azure/AzureML-Containers"}),"https://github.com/Azure/AzureML-Containers")),Object(o.b)("li",{parentName:"ul"},"The dockerfile references ",Object(o.b)("inlineCode",{parentName:"li"},"mutated_conda_dependencies.yml")," to build the Python environment via Conda.")),Object(o.b)("p",null,"Get the contents of ",Object(o.b)("inlineCode",{parentName:"p"},"mutated_conda_dependencies.yml")," from the environment:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"print(env.python.conda_dependencies.serialize_to_string())\n")),Object(o.b)("p",null,"Which looks like"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash",metastring:'title="mutated_conda_dependencies.yml"',title:'"mutated_conda_dependencies.yml"'}),"channels:\n    - defaults\n    - pytorch\ndependencies:\n    - python=3.7\n    - pytorch\n    - torchvision\nname: azureml_7459a71437df47401c6a369f49fbbdb6\n")))}s.isMDXComponent=!0}}]);