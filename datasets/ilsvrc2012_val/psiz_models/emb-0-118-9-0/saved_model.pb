??
?
?

B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8?v
f
	kl_annealVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	kl_anneal
_
kl_anneal/Read/ReadVariableOpReadVariableOp	kl_anneal*
_output_shapes
: *
dtype0
d
locVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ц	*
shared_nameloc
]
loc/Read/ReadVariableOpReadVariableOploc* 
_output_shapes
:
ц	*
dtype0
?
untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ц	*$
shared_nameuntransformed_scale
}
'untransformed_scale/Read/ReadVariableOpReadVariableOpuntransformed_scale* 
_output_shapes
:
ц	*
dtype0
Z
rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namerho
S
rho/Read/ReadVariableOpReadVariableOprho*
_output_shapes
: *
dtype0
?
rank/distance_based/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namerank/distance_based/w
{
)rank/distance_based/w/Read/ReadVariableOpReadVariableOprank/distance_based/w*
_output_shapes
:	*
dtype0
Z
tauVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametau
S
tau/Read/ReadVariableOpReadVariableOptau*
_output_shapes
: *
dtype0
^
gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namegamma
W
gamma/Read/ReadVariableOpReadVariableOpgamma*
_output_shapes
: *
dtype0
\
betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta
U
beta/Read/ReadVariableOpReadVariableOpbeta*
_output_shapes
: *
dtype0
f
loc_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameloc_1
_
loc_1/Read/ReadVariableOpReadVariableOploc_1*
_output_shapes

:*
dtype0
?
untransformed_scale_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameuntransformed_scale_1

)untransformed_scale_1/Read/ReadVariableOpReadVariableOpuntransformed_scale_1*
_output_shapes

:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
{
stimuli

kernel
behavior

_use_group
	optimizer

signatures
#_self_saveable_object_factories
N
	posterior
		prior

	kl_anneal
#_self_saveable_object_factories
C
distance

similarity
#_self_saveable_object_factories
%
#_self_saveable_object_factories
 
 
 
 
W
loc
untransformed_scale

embeddings
#_self_saveable_object_factories
5

_embedding
#_self_saveable_object_factories
KI
VARIABLE_VALUE	kl_anneal,stimuli/kl_anneal/.ATTRIBUTES/VARIABLE_VALUE
 
5
rho
w
#_self_saveable_object_factories
C
tau
	gamma
beta
#_self_saveable_object_factories
 
 
IG
VARIABLE_VALUEloc0stimuli/posterior/loc/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEuntransformed_scale@stimuli/posterior/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE
L
_distribution
_graph_parents
#_self_saveable_object_factories
 
W
 loc
!untransformed_scale
"
embeddings
##_self_saveable_object_factories
 
GE
VARIABLE_VALUErho.kernel/distance/rho/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUErank/distance_based/w,kernel/distance/w/.ATTRIBUTES/VARIABLE_VALUE
 
IG
VARIABLE_VALUEtau0kernel/similarity/tau/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgamma2kernel/similarity/gamma/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEbeta1kernel/similarity/beta/.ATTRIBUTES/VARIABLE_VALUE
 
O
_loc

$_scale
%_graph_parents
#&_self_saveable_object_factories
 
 
RP
VARIABLE_VALUEloc_17stimuli/prior/_embedding/loc/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEuntransformed_scale_1Gstimuli/prior/_embedding/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE
L
'_distribution
(_graph_parents
#)_self_saveable_object_factories
 
@
_pretransformed_input
#*_self_saveable_object_factories
 
 
O
 _loc

+_scale
,_graph_parents
#-_self_saveable_object_factories
 
 
 
@
!_pretransformed_input
#._self_saveable_object_factories
 
 
 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenamekl_anneal/Read/ReadVariableOploc/Read/ReadVariableOp'untransformed_scale/Read/ReadVariableOprho/Read/ReadVariableOp)rank/distance_based/w/Read/ReadVariableOptau/Read/ReadVariableOpgamma/Read/ReadVariableOpbeta/Read/ReadVariableOploc_1/Read/ReadVariableOp)untransformed_scale_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_77472
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	kl_anneallocuntransformed_scalerhorank/distance_based/wtaugammabetaloc_1untransformed_scale_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_77512?M
? 
?
__inference__traced_save_77472
file_prefix(
$savev2_kl_anneal_read_readvariableop"
savev2_loc_read_readvariableop2
.savev2_untransformed_scale_read_readvariableop"
savev2_rho_read_readvariableop4
0savev2_rank_distance_based_w_read_readvariableop"
savev2_tau_read_readvariableop$
 savev2_gamma_read_readvariableop#
savev2_beta_read_readvariableop$
 savev2_loc_1_read_readvariableop4
0savev2_untransformed_scale_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,stimuli/kl_anneal/.ATTRIBUTES/VARIABLE_VALUEB0stimuli/posterior/loc/.ATTRIBUTES/VARIABLE_VALUEB@stimuli/posterior/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB.kernel/distance/rho/.ATTRIBUTES/VARIABLE_VALUEB,kernel/distance/w/.ATTRIBUTES/VARIABLE_VALUEB0kernel/similarity/tau/.ATTRIBUTES/VARIABLE_VALUEB2kernel/similarity/gamma/.ATTRIBUTES/VARIABLE_VALUEB1kernel/similarity/beta/.ATTRIBUTES/VARIABLE_VALUEB7stimuli/prior/_embedding/loc/.ATTRIBUTES/VARIABLE_VALUEBGstimuli/prior/_embedding/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_kl_anneal_read_readvariableopsavev2_loc_read_readvariableop.savev2_untransformed_scale_read_readvariableopsavev2_rho_read_readvariableop0savev2_rank_distance_based_w_read_readvariableopsavev2_tau_read_readvariableop savev2_gamma_read_readvariableopsavev2_beta_read_readvariableop savev2_loc_1_read_readvariableop0savev2_untransformed_scale_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*S
_input_shapesB
@: : :
ц	:
ц	: :	: : : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :&"
 
_output_shapes
:
ц	:&"
 
_output_shapes
:
ц	:

_output_shapes
: : 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

::$
 

_output_shapes

::

_output_shapes
: 
?,
?
!__inference__traced_restore_77512
file_prefix
assignvariableop_kl_anneal
assignvariableop_1_loc*
&assignvariableop_2_untransformed_scale
assignvariableop_3_rho,
(assignvariableop_4_rank_distance_based_w
assignvariableop_5_tau
assignvariableop_6_gamma
assignvariableop_7_beta
assignvariableop_8_loc_1,
(assignvariableop_9_untransformed_scale_1
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B,stimuli/kl_anneal/.ATTRIBUTES/VARIABLE_VALUEB0stimuli/posterior/loc/.ATTRIBUTES/VARIABLE_VALUEB@stimuli/posterior/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB.kernel/distance/rho/.ATTRIBUTES/VARIABLE_VALUEB,kernel/distance/w/.ATTRIBUTES/VARIABLE_VALUEB0kernel/similarity/tau/.ATTRIBUTES/VARIABLE_VALUEB2kernel/similarity/gamma/.ATTRIBUTES/VARIABLE_VALUEB1kernel/similarity/beta/.ATTRIBUTES/VARIABLE_VALUEB7stimuli/prior/_embedding/loc/.ATTRIBUTES/VARIABLE_VALUEBGstimuli/prior/_embedding/untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_kl_annealIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_locIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_untransformed_scaleIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_rhoIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rank_distance_based_wIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_tauIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_loc_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_untransformed_scale_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?8
stimuli

kernel
behavior

_use_group
	optimizer

signatures
#_self_saveable_object_factories"?7
_tf_keras_model?7{"class_name": "psiz.keras.models>Rank", "name": "rank", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "rank", "class_name": "Rank", "psiz_version": "0.5.0", "n_sample": 100, "layers": {"stimuli": {"class_name": "psiz.keras.layers>EmbeddingVariational", "config": {"name": "embedding_variational", "trainable": true, "dtype": "float32", "posterior": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "mask_zero": true, "input_length": 1, "loc_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": true, "scale_trainable": true}}, "prior": {"class_name": "psiz.keras.layers>EmbeddingShared", "config": {"name": "embedding_shared", "trainable": true, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "embedding": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 1, "output_dim": 1, "mask_zero": false, "input_length": 1, "loc_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": false, "scale_trainable": true}}, "mask_zero": true}}, "kl_weight": 4.165104752384522e-05, "kl_use_exact": false, "kl_n_sample": 100}}, "kernel": {"class_name": "psiz.keras.layers>DistanceBased", "config": {"name": "distance_based", "trainable": true, "dtype": "float32", "distance": {"class_name": "psiz.keras.layers>Minkowski", "config": {"name": "minkowski", "trainable": true, "dtype": "float32", "rho_initializer": {"class_name": "Constant", "config": {"value": 2.0}}, "w_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "rho_regularizer": null, "w_regularizer": null, "rho_constraint": {"class_name": "psiz.keras.constraints>GreaterEqualThan", "config": {"min_value": 1.0}}, "w_constraint": {"class_name": "NonNeg", "config": {}}, "rho_trainable": false, "w_trainable": false}}, "similarity": {"class_name": "psiz.keras.layers>ExponentialSimilarity", "config": {"name": "exponential_similarity", "trainable": true, "dtype": "float32", "fit_tau": true, "fit_gamma": true, "fit_beta": false, "tau_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "gamma_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "beta_initializer": {"class_name": "Constant", "config": {"value": 10.0}}}}}}, "behavior": {"class_name": "psiz.keras.layers>RankBehavior", "config": {"name": "rank_behavior", "trainable": true, "dtype": "float32"}}}, "use_group_stimuli": false, "use_group_kernel": false, "use_group_behavior": false}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Rank", "config": {"name": "rank", "class_name": "Rank", "psiz_version": "0.5.0", "n_sample": 100, "layers": {"stimuli": {"class_name": "psiz.keras.layers>EmbeddingVariational", "config": {"name": "embedding_variational", "trainable": true, "dtype": "float32", "posterior": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "mask_zero": true, "input_length": 1, "loc_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": true, "scale_trainable": true}}, "prior": {"class_name": "psiz.keras.layers>EmbeddingShared", "config": {"name": "embedding_shared", "trainable": true, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "embedding": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 1, "output_dim": 1, "mask_zero": false, "input_length": 1, "loc_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": false, "scale_trainable": true}}, "mask_zero": true}}, "kl_weight": 4.165104752384522e-05, "kl_use_exact": false, "kl_n_sample": 100}}, "kernel": {"class_name": "psiz.keras.layers>DistanceBased", "config": {"name": "distance_based", "trainable": true, "dtype": "float32", "distance": {"class_name": "psiz.keras.layers>Minkowski", "config": {"name": "minkowski", "trainable": true, "dtype": "float32", "rho_initializer": {"class_name": "Constant", "config": {"value": 2.0}}, "w_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "rho_regularizer": null, "w_regularizer": null, "rho_constraint": {"class_name": "psiz.keras.constraints>GreaterEqualThan", "config": {"min_value": 1.0}}, "w_constraint": {"class_name": "NonNeg", "config": {}}, "rho_trainable": false, "w_trainable": false}}, "similarity": {"class_name": "psiz.keras.layers>ExponentialSimilarity", "config": {"name": "exponential_similarity", "trainable": true, "dtype": "float32", "fit_tau": true, "fit_gamma": true, "fit_beta": false, "tau_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "gamma_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "beta_initializer": {"class_name": "Constant", "config": {"value": 10.0}}}}}}, "behavior": {"class_name": "psiz.keras.layers>RankBehavior", "config": {"name": "rank_behavior", "trainable": true, "dtype": "float32"}}}, "use_group_stimuli": false, "use_group_kernel": false, "use_group_behavior": false}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": null, "weighted_metrics": [[{"class_name": "CategoricalCrossentropy", "config": {"name": "cce", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}]], "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	posterior
		prior

	kl_anneal
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>EmbeddingVariational", "name": "embedding_variational", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_variational", "trainable": true, "dtype": "float32", "posterior": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "mask_zero": true, "input_length": 1, "loc_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": true, "scale_trainable": true}}, "prior": {"class_name": "psiz.keras.layers>EmbeddingShared", "config": {"name": "embedding_shared", "trainable": true, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "embedding": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 1, "output_dim": 1, "mask_zero": false, "input_length": 1, "loc_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": false, "scale_trainable": true}}, "mask_zero": true}}, "kl_weight": 4.165104752384522e-05, "kl_use_exact": false, "kl_n_sample": 100}, "build_input_shape": [null, null, null]}
?
distance

similarity
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>DistanceBased", "name": "distance_based", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "distance_based", "trainable": true, "dtype": "float32", "distance": {"class_name": "psiz.keras.layers>Minkowski", "config": {"name": "minkowski", "trainable": true, "dtype": "float32", "rho_initializer": {"class_name": "Constant", "config": {"value": 2.0}}, "w_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "rho_regularizer": null, "w_regularizer": null, "rho_constraint": {"class_name": "psiz.keras.constraints>GreaterEqualThan", "config": {"min_value": 1.0}}, "w_constraint": {"class_name": "NonNeg", "config": {}}, "rho_trainable": false, "w_trainable": false}}, "similarity": {"class_name": "psiz.keras.layers>ExponentialSimilarity", "config": {"name": "exponential_similarity", "trainable": true, "dtype": "float32", "fit_tau": true, "fit_gamma": true, "fit_beta": false, "tau_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "gamma_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "beta_initializer": {"class_name": "Constant", "config": {"value": 10.0}}}}}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 1, 56, 9]}, {"class_name": "TensorShape", "items": [null, null, null, 56, 9]}]}
?
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>RankBehavior", "name": "rank_behavior", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rank_behavior", "trainable": true, "dtype": "float32"}}
 "
trackable_dict_wrapper
"
	optimizer
"
signature_map
 "
trackable_dict_wrapper
?
loc
untransformed_scale

embeddings
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "name": "embedding_normal_diag", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_normal_diag", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "mask_zero": true, "input_length": 1, "loc_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": true, "scale_trainable": true}}
?

_embedding
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>EmbeddingShared", "name": "embedding_shared", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_shared", "trainable": true, "dtype": "float32", "input_dim": 50001, "output_dim": 9, "embedding": {"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "config": {"name": "embedding_normal_diag_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 1, "output_dim": 1, "mask_zero": false, "input_length": 1, "loc_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": false, "scale_trainable": true}}, "mask_zero": true}, "build_input_shape": [null, null, null]}
: 2	kl_anneal
 "
trackable_dict_wrapper
?
rho
w
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>Minkowski", "name": "minkowski", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "minkowski", "trainable": true, "dtype": "float32", "rho_initializer": {"class_name": "Constant", "config": {"value": 2.0}}, "w_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "rho_regularizer": null, "w_regularizer": null, "rho_constraint": {"class_name": "psiz.keras.constraints>GreaterEqualThan", "config": {"min_value": 1.0}}, "w_constraint": {"class_name": "NonNeg", "config": {}}, "rho_trainable": false, "w_trainable": false}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, null, 1, 56, 9]}, {"class_name": "TensorShape", "items": [null, null, null, 56, 9]}]}
?
tau
	gamma
beta
#_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>ExponentialSimilarity", "name": "exponential_similarity", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "exponential_similarity", "trainable": true, "dtype": "float32", "fit_tau": true, "fit_gamma": true, "fit_beta": false, "tau_initializer": {"class_name": "Constant", "config": {"value": 1.0}}, "gamma_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "beta_initializer": {"class_name": "Constant", "config": {"value": 10.0}}}}
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
:
ц	2loc
':%
ц	2untransformed_scale
j
_distribution
_graph_parents
#_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
?
 loc
!untransformed_scale
"
embeddings
##_self_saveable_object_factories"?
_tf_keras_layer?{"class_name": "psiz.keras.layers>EmbeddingNormalDiag", "name": "embedding_normal_diag_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_normal_diag_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 1, "output_dim": 1, "mask_zero": false, "input_length": 1, "loc_initializer": {"class_name": "Constant", "config": {"value": 0.0}}, "scale_initializer": {"class_name": "Constant", "config": {"value": -1.8952964544296265}}, "loc_regularizer": null, "scale_regularizer": null, "loc_constraint": null, "scale_constraint": null, "loc_trainable": false, "scale_trainable": true}}
 "
trackable_dict_wrapper
:	 2rho
!:	2rank/distance_based/w
 "
trackable_dict_wrapper
:	 2tau
: 2gamma
:
 2beta
 "
trackable_dict_wrapper
m
_loc

$_scale
%_graph_parents
#&_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:2loc
%:#2untransformed_scale
j
'_distribution
(_graph_parents
#)_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
^
_pretransformed_input
#*_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
m
 _loc

+_scale
,_graph_parents
#-_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
^
!_pretransformed_input
#._self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper