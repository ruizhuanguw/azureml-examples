(window.webpackJsonp=window.webpackJsonp||[]).push([[18],{106:function(e,t,n){"use strict";n.d(t,"a",(function(){return s})),n.d(t,"b",(function(){return d}));var r=n(0),o=n.n(r);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function c(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?c(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var l=o.a.createContext({}),u=function(e){var t=o.a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},s=function(e){var t=u(e.components);return o.a.createElement(l.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},m=o.a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,a=e.originalType,c=e.parentName,l=p(e,["components","mdxType","originalType","parentName"]),s=u(n),m=r,d=s["".concat(c,".").concat(m)]||s[m]||b[m]||a;return n?o.a.createElement(d,i(i({ref:t},l),{},{components:n})):o.a.createElement(d,i({ref:t},l))}));function d(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var a=n.length,c=new Array(a);c[0]=m;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i.mdxType="string"==typeof e?e:r,c[1]=i;for(var l=2;l<a;l++)c[l]=n[l];return o.a.createElement.apply(null,c)}return o.a.createElement.apply(null,n)}m.displayName="MDXCreateElement"},76:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return c})),n.d(t,"metadata",(function(){return i})),n.d(t,"rightToc",(function(){return p})),n.d(t,"default",(function(){return u}));var r=n(2),o=n(6),a=(n(0),n(106)),c={},i={unversionedId:"templates",id:"templates",isDocsHomePage:!1,title:"templates",description:"Templates",source:"@site/docs/templates.md",slug:"/templates",permalink:"/AzureML-CheatSheet/docs/templates",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/docs/templates.md",version:"current"},p=[{value:"Introduction",id:"introduction",children:[]},{value:"Project Templates",id:"project-templates",children:[{value:"ScriptRunConfig",id:"scriptrunconfig",children:[]}]}],l={rightToc:p};function u(e){var t=e.components,n=Object(o.a)(e,["components"]);return Object(a.b)("wrapper",Object(r.a)({},l,n,{components:t,mdxType:"MDXLayout"}),Object(a.b)("h1",{id:"templates"},"Templates"),Object(a.b)("h2",{id:"introduction"},"Introduction"),Object(a.b)("p",null,"Cookiecutter is a simple command-line tool that allows you to quickly create\nnew projects from pre-defined templates. Let's see it in action!"),Object(a.b)("p",null,"First go ahead and get cookiecutter using your environment manager of choice,\nfor example:"),Object(a.b)("pre",null,Object(a.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"pip install cookiecutter\n")),Object(a.b)("p",null,"Then give this repo a home"),Object(a.b)("pre",null,Object(a.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"cd ~/repos # or wherever your repos call home :-)\ngit clone <this-repo>\n")),Object(a.b)("p",null,"To create a new project from the ",Object(a.b)("inlineCode",{parentName:"p"},"ScriptRunConfig")," template for example, simply\nrun"),Object(a.b)("pre",null,Object(a.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"cookiecutter path/to/cheatsheet/repo/templates/ScriptRunConfig\n")),Object(a.b)("p",null,"See ",Object(a.b)("a",Object(r.a)({parentName:"p"},{href:"#ScriptRunConfig"}),"ScriptRunConfig")," for more details for this template."),Object(a.b)("h2",{id:"project-templates"},"Project Templates"),Object(a.b)("ul",null,Object(a.b)("li",{parentName:"ul"},"ScriptRunConfig: Create a project to run a script in AML making use of the\nScriptRunConfig class. This template is well suited for smaller projects and\nis especially handy for testing.")),Object(a.b)("h3",{id:"scriptrunconfig"},"ScriptRunConfig"),Object(a.b)("p",null,Object(a.b)("a",Object(r.a)({parentName:"p"},{href:"https://cookiecutter.readthedocs.io/en/1.7.2/README.html"}),"Cookiecutter"),"\ntemplate for setting up an AML\n",Object(a.b)("a",Object(r.a)({parentName:"p"},{href:"https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py"}),"ScriptRunConfig"),"\nused to run your script in Azure."),Object(a.b)("h4",{id:"usage"},"Usage"),Object(a.b)("p",null,"Run the cookiecutter command"),Object(a.b)("pre",null,Object(a.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"cookiecutter <path/to/cookiecutter/templates>/ScriptRunConfig\n")),Object(a.b)("p",null,"to create a new ",Object(a.b)("inlineCode",{parentName:"p"},"ScriptRunConfig")," project."),Object(a.b)("p",null,Object(a.b)("strong",{parentName:"p"},"Note.")," Install with ",Object(a.b)("inlineCode",{parentName:"p"},"pip install cookiecutter")," (see\n",Object(a.b)("a",Object(r.a)({parentName:"p"},{href:"https://cookiecutter.readthedocs.io/en/1.7.2/installation.html"}),"cookiecutter docs"),"\nfor more installation options)"),Object(a.b)("p",null,"You will be prompted for the following:"),Object(a.b)("ul",null,Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"directory_name"),': The desired name of the directory (default:\n"aml-src-script")'),Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"script_name"),': The name of the python script to be run in Azure (default:\n"script")'),Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"subscription_id"),": Your Azure Subscription ID"),Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"resource_group"),": Your Azure resource group name"),Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"workspace_name"),": Your Azure ML workspace name"),Object(a.b)("li",{parentName:"ul"},Object(a.b)("inlineCode",{parentName:"li"},"compute_target_name"),': The name of the Azure ML compute target to run the\nscript on (default: "local", will run on your box)')),Object(a.b)("p",null,"Cookiecutter creates a new project with the following layout."),Object(a.b)("pre",null,Object(a.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"{directory_name}/\n    {script_name}.py      # the script you want to run in the cloud\n    run.py                # wraps your script in ScriptRunConfig to send to Azure\n    config.json           # your Azure ML metadata\n    readme.md             # this readme file!\n")),Object(a.b)("p",null,"See\n",Object(a.b)("a",Object(r.a)({parentName:"p"},{href:"https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py"}),"ScriptRunConfig"),"\nfor more options and details on configuring runs."))}u.isMDXComponent=!0}}]);