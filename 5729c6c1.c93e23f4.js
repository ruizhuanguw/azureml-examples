(window.webpackJsonp=window.webpackJsonp||[]).push([[19],{106:function(e,t,a){"use strict";a.d(t,"a",(function(){return l})),a.d(t,"b",(function(){return m}));var r=a(0),n=a.n(r);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function c(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function s(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?c(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):c(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function i(e,t){if(null==e)return{};var a,r,n=function(e,t){if(null==e)return{};var a,r,n={},o=Object.keys(e);for(r=0;r<o.length;r++)a=o[r],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)a=o[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var d=n.a.createContext({}),p=function(e){var t=n.a.useContext(d),a=t;return e&&(a="function"==typeof e?e(t):s(s({},t),e)),a},l=function(e){var t=p(e.components);return n.a.createElement(d.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.a.createElement(n.a.Fragment,{},t)}},b=n.a.forwardRef((function(e,t){var a=e.components,r=e.mdxType,o=e.originalType,c=e.parentName,d=i(e,["components","mdxType","originalType","parentName"]),l=p(a),b=r,m=l["".concat(c,".").concat(b)]||l[b]||u[b]||o;return a?n.a.createElement(m,s(s({ref:t},d),{},{components:a})):n.a.createElement(m,s({ref:t},d))}));function m(e,t){var a=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=a.length,c=new Array(o);c[0]=b;var s={};for(var i in t)hasOwnProperty.call(t,i)&&(s[i]=t[i]);s.originalType=e,s.mdxType="string"==typeof e?e:r,c[1]=s;for(var d=2;d<o;d++)c[d]=a[d];return n.a.createElement.apply(null,c)}return n.a.createElement.apply(null,a)}b.displayName="MDXCreateElement"},77:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return c})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return i})),a.d(t,"default",(function(){return p}));var r=a(2),n=a(6),o=(a(0),a(106)),c={title:"Datastore"},s={unversionedId:"datastore",id:"datastore",isDocsHomePage:!1,title:"Datastore",description:"Get Datastore",source:"@site/docs/datastore.md",slug:"/datastore",permalink:"/AzureML-CheatSheet/docs/datastore",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/docs/datastore.md",version:"current"},i=[{value:"Get Datastore",id:"get-datastore",children:[{value:"Default datastore",id:"default-datastore",children:[]},{value:"Registered datastores",id:"registered-datastores",children:[]}]},{value:"Upload to Datastore",id:"upload-to-datastore",children:[{value:"Via SDK",id:"via-sdk",children:[]},{value:"Via Storage Explorer",id:"via-storage-explorer",children:[]}]},{value:"Read from Datastore",id:"read-from-datastore",children:[{value:"DataReference",id:"datareference",children:[]}]}],d={rightToc:i};function p(e){var t=e.components,a=Object(n.a)(e,["components"]);return Object(o.b)("wrapper",Object(r.a)({},d,a,{components:t,mdxType:"MDXLayout"}),Object(o.b)("h2",{id:"get-datastore"},"Get Datastore"),Object(o.b)("h3",{id:"default-datastore"},"Default datastore"),Object(o.b)("p",null,"Each workspace comes with a default datastore."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"datastore = ws.get_default_datastore()\n")),Object(o.b)("h3",{id:"registered-datastores"},"Registered datastores"),Object(o.b)("p",null,"Connect to, or create, a datastore backed by one of the multiple data-storage options\nthat Azure provides."),Object(o.b)("h4",{id:"register-a-new-datastore"},"Register a new datastore"),Object(o.b)("p",null,"To register a store via a SAS token:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),'datastores = Datastore.register_azure_blob_container(\n    workspace=ws,\n    datastore_name="<datastore-name>",\n    container_name="<container-name>",\n    account_name="<account-name>",\n    sas_token="<sas-token>",\n)\n')),Object(o.b)("p",null,"For more ways authentication options and for different underlying storage see\nthe AML documentation on\n",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.datastore(class)?view=azure-ml-py"}),"Datastores"),"."),Object(o.b)("h4",{id:"connect-to-registered-datastore"},"Connect to registered datastore"),Object(o.b)("p",null,"Any datastore that is registered to workspace can be accessed by name."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),'from azureml.core import Datastore\ndatastore = Datastore.get(ws, "<name-of-registered-datastore>")\n')),Object(o.b)("h2",{id:"upload-to-datastore"},"Upload to Datastore"),Object(o.b)("h3",{id:"via-sdk"},"Via SDK"),Object(o.b)("p",null,"The datastore provides APIs for data upload:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"datastore.upload(src_dir='./data', target_path='<path/on/datastore>', overwrite=True)\n")),Object(o.b)("h3",{id:"via-storage-explorer"},"Via Storage Explorer"),Object(o.b)("p",null,"Azure Storage Explorer is free tool to easily manage your Azure cloud storage\nresources from Windows, macOS, or Linux. Download it from ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://azure.microsoft.com/features/storage-explorer/"}),"here"),"."),Object(o.b)("p",null,"Azure Storage Explorer gives you a (graphical) file exporer, so you can literally drag and drop\nfiles into your Datastores."),Object(o.b)("h4",{id:"working-with-the-default-datastore"},"Working with the default datastore"),Object(o.b)("p",null,"Each workspace comes with its own datastore (e.g. ",Object(o.b)("inlineCode",{parentName:"p"},"ws.get_default_datastore"),"). Visit ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://portal.azure.com"}),"https://portal.azure.com"),"\nand locate your workspace's resource group and find the storage account."),Object(o.b)("h2",{id:"read-from-datastore"},"Read from Datastore"),Object(o.b)("p",null,"Reference data in a ",Object(o.b)("inlineCode",{parentName:"p"},"Datastore")," in your code, for example to use in a remote setting."),Object(o.b)("h3",{id:"datareference"},"DataReference"),Object(o.b)("p",null,"First, connect to your basic assets: ",Object(o.b)("inlineCode",{parentName:"p"},"Workspace"),", ",Object(o.b)("inlineCode",{parentName:"p"},"ComputeTarget")," and ",Object(o.b)("inlineCode",{parentName:"p"},"Datastore"),"."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"from azureml.core import Workspace\nws: Workspace = Workspace.from_config()\ncompute_target: ComputeTarget = ws.compute_targets['<compute-target-name>']\nds: Datastore = ws.get_default_datastore()\n")),Object(o.b)("p",null,"Create a ",Object(o.b)("inlineCode",{parentName:"p"},"DataReference"),", either as mount:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"data_ref = ds.path('<path/on/datastore>').as_mount()\n")),Object(o.b)("p",null,"or as download:"),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"data_ref = ds.path('<path/on/datastore>').as_download()\n")),Object(o.b)("h4",{id:"consume-datareference-in-scriptrunconfig"},"Consume DataReference in ScriptRunConfig"),Object(o.b)("p",null,"Add this DataReference to a ScriptRunConfig as follows."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"config = ScriptRunConfig(\n    source_directory='.',\n    script='script.py',\n    arguments=[str(data_ref)],               # returns environment variable $AZUREML_DATAREFERENCE_example_data\n    compute_target=compute_target,\n)\n\nconfig.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()\n")),Object(o.b)("p",null,"The command-line argument ",Object(o.b)("inlineCode",{parentName:"p"},"str(data_ref)")," returns the environment variable ",Object(o.b)("inlineCode",{parentName:"p"},"$AZUREML_DATAREFERENCE_example_data"),".\nFinally, ",Object(o.b)("inlineCode",{parentName:"p"},"data_ref.to_config()")," instructs the run to mount the data to the compute target and to assign the\nabove environment variable appropriately."),Object(o.b)("h4",{id:"without-specifying-argument"},"Without specifying argument"),Object(o.b)("p",null,"Specify a ",Object(o.b)("inlineCode",{parentName:"p"},"path_on_compute")," to reference your data without the need for command-line arguments."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),"data_ref = ds.path('<path/on/datastore>').as_mount()\ndata_ref.path_on_compute = '/tmp/data'\n\nconfig = ScriptRunConfig(\n    source_directory='.',\n    script='script.py',\n    compute_target=compute_target,\n)\n\nconfig.run_config.data_references[data_ref.data_reference_name] = data_ref.to_config()\n")))}p.isMDXComponent=!0}}]);